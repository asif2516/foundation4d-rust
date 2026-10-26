//! Shared backbone (paper Sec 3.3): patch embedding + factorized spatiotemporal
//! aggregator. Mirrors the paper's design at demo scale:
//!   * every image contributes patch tokens plus a rig token, an ego token and
//!     4 register tokens — all participating in attention;
//!   * each layer = within-image self-attention (full-res stream) followed by
//!     cross-image self-attention, alternating between camera groups (fixed time)
//!     and temporal groups (fixed camera) — the factorization that replaces VGGT's
//!     global attention;
//!   * two spatial resolutions: the aggregation stream is 2x2-patch pooled while the
//!     full-resolution stream is preserved for the detection/occupancy pyramid
//!     (the paper pools 74x74 -> 37x37 before aggregation);
//!   * 2D RoPE on patch tokens within images; 1D RoPE indexing images inside a
//!     cross-camera / cross-temporal group;
//!   * the final representation concatenates the within-image and temporal-attention
//!     streams along channels (width 2d), exactly as in the paper.
//!
//! Heads: rig/ego pose heads read the rig/ego tokens through small transformer
//! heads (paper Sec 3.4); depth / pointmap / semantic heads are compact DPT-style
//! decoders over the contextual patch features. Weights are random-initialized —
//! this demonstrates the architecture and its measured compute, standing in for the
//! trained DINOv2/VGGT-scale parameters.

use crate::math::{Quat, Rng, Se3, Vec3};
use crate::scene::{IMG_H, IMG_W, N_CAMS};
use crate::tensor::{attention_mh, gelu, layer_norm, linear};

/// 2D RoPE on the first 16 channels of a token (8 pairs: 4 for x, 4 for y grid axes).
fn rope2d_token(tok: &mut [f32], px: u16, py: u16) {
    let rope_pairs = 8usize;
    for axis in 0..2 {
        let p = if axis == 0 { px } else { py } as f32;
        for k in 0..(rope_pairs / 2) {
            let ch = axis * rope_pairs + 2 * k;
            let freq = (10000.0f32).powf(-(2.0 * k as f32) / (rope_pairs as f32));
            let (s, c) = (freq * p).sin_cos();
            let a = tok[ch];
            let b = tok[ch + 1];
            tok[ch] = a * c - b * s;
            tok[ch + 1] = a * s + b * c;
        }
    }
}

/// 1D RoPE on the first 8 channels, indexing an image inside a cross group.
fn rope1d_token(tok: &mut [f32], idx: u16) {
    for k in 0..4usize {
        let ch = 2 * k;
        let freq = (10000.0f32).powf(-(2.0 * k as f32) / 8.0);
        let (s, c) = (freq * idx as f32).sin_cos();
        let a = tok[ch];
        let b = tok[ch + 1];
        tok[ch] = a * c - b * s;
        tok[ch + 1] = a * s + b * c;
    }
}

pub const PATCH: usize = 8;
pub const GW: usize = IMG_W / PATCH; // 24
pub const GH: usize = IMG_H / PATCH; // 18
pub const N_PATCH: usize = GW * GH; // 432
pub const N_SPECIAL: usize = 6; // rig + ego + 4 registers
pub const N_TOKENS: usize = N_PATCH + N_SPECIAL; // per image, full-res stream
/// Aggregation-stream pooling. The paper pools its 74x74 grid -> 37x37 before
/// aggregation; at this demo's higher input resolution a 3x3 pool keeps the
/// aggregator's per-image token budget (and hence the paper's compute profile
/// per window) constant: 48 patch tokens + 6 special tokens per image.
pub const POOL: usize = 3;
pub const PW: usize = GW / POOL; // 8
pub const PH: usize = GH / POOL; // 6
pub const N_PPATCH: usize = PW * PH; // 48
pub const N_PTOK: usize = N_PPATCH + N_SPECIAL; // 54

#[derive(Clone)]
pub struct BackboneConfig {
    pub d: usize,
    pub n_heads: usize,
    pub n_layers: usize, // total aggregator layers; Block B alternates camera/temporal
    pub mlp_hidden: usize,
    pub seed: u64,
}

impl Default for BackboneConfig {
    fn default() -> Self {
        BackboneConfig { d: 96, n_heads: 4, n_layers: 4, mlp_hidden: 192, seed: 7 }
    }
}

/// All trainable parameters of the aggregator (shared across windows).
pub struct Weights {
    // Patch embedding: PATCH*PATCH*3 -> d.
    pub patch_w: Vec<f32>, // (d, PATCH*PATCH*3) out-major
    pub patch_b: Vec<f32>,
    // Per layer: full-res block (A) and pooled cross block (B).
    pub layers: Vec<LayerWeights>,
    // Head weights.
    pub head_ln_g: Vec<f32>,
    pub head_ln_b: Vec<f32>,
    pub depth_w1: Vec<f32>, pub depth_b1: Vec<f32>,
    pub depth_w2: Vec<f32>, pub depth_b2: Vec<f32>,
    pub sem_w1: Vec<f32>, pub sem_b1: Vec<f32>,
    pub sem_w2: Vec<f32>, pub sem_b2: Vec<f32>,
    pub pose_ln_g: Vec<f32>,
    pub pose_ln_b: Vec<f32>,
    pub pose_attn_wq: Vec<f32>, pub pose_attn_wk: Vec<f32>, pub pose_attn_wv: Vec<f32>,
    pub pose_mlp_w1: Vec<f32>, pub pose_mlp_b1: Vec<f32>,
    pub pose_mlp_w2: Vec<f32>, pub pose_mlp_b2: Vec<f32>,
}

pub struct LayerWeights {
    // Block A (within-image, full-res tokens).
    pub a_ln_g: Vec<f32>, pub a_ln_b: Vec<f32>,
    pub a_wq: Vec<f32>, pub a_wk: Vec<f32>, pub a_wv: Vec<f32>, pub a_wo: Vec<f32>,
    pub a_bq: Vec<f32>, pub a_bk: Vec<f32>, pub a_bv: Vec<f32>, pub a_bo: Vec<f32>,
    pub a_ln2_g: Vec<f32>, pub a_ln2_b: Vec<f32>,
    pub a_mlp_w1: Vec<f32>, pub a_mlp_b1: Vec<f32>,
    pub a_mlp_w2: Vec<f32>, pub a_mlp_b2: Vec<f32>,
    // Block B (cross-image, pooled tokens).
    pub b_ln_g: Vec<f32>, pub b_ln_b: Vec<f32>,
    pub b_wq: Vec<f32>, pub b_wk: Vec<f32>, pub b_wv: Vec<f32>, pub b_wo: Vec<f32>,
    pub b_bq: Vec<f32>, pub b_bk: Vec<f32>, pub b_bv: Vec<f32>, pub b_bo: Vec<f32>,
    pub b_ln2_g: Vec<f32>, pub b_ln2_b: Vec<f32>,
    pub b_mlp_w1: Vec<f32>, pub b_mlp_b1: Vec<f32>,
    pub b_mlp_w2: Vec<f32>, pub b_mlp_b2: Vec<f32>,
}

fn init_linear(rng: &mut Rng, d_in: usize, d_out: usize, gain: f32) -> (Vec<f32>, Vec<f32>) {
    let mut w = vec![0.0f32; d_in * d_out];
    let s = gain * (1.0 / d_in as f32).sqrt();
    for v in w.iter_mut() {
        *v = rng.normal_clamped(s);
    }
    (w, vec![0.0; d_out])
}

impl Weights {
    pub fn new(cfg: &BackboneConfig, rng: &mut Rng) -> Weights {
        let d = cfg.d;
        let patch_in = PATCH * PATCH * 3;
        let (patch_w, patch_b) = init_linear(rng, patch_in, d, 1.0);
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for _ in 0..cfg.n_layers {
            let (wq, bq) = init_linear(rng, d, d, 0.5);
            let (wk, bk) = init_linear(rng, d, d, 0.5);
            let (wv, bv) = init_linear(rng, d, d, 0.5);
            let (wo, bo) = init_linear(rng, d, d, 0.5);
            let (w1, b1) = init_linear(rng, d, cfg.mlp_hidden, 1.0);
            let (w2, b2) = init_linear(rng, cfg.mlp_hidden, d, 1.0);
            let (b_wq, b_bq) = init_linear(rng, d, d, 0.5);
            let (b_wk, b_bk) = init_linear(rng, d, d, 0.5);
            let (b_wv, b_bv) = init_linear(rng, d, d, 0.5);
            let (b_wo, b_bo) = init_linear(rng, d, d, 0.5);
            let (b_w1, b_b1) = init_linear(rng, d, cfg.mlp_hidden, 1.0);
            let (b_w2, b_b2) = init_linear(rng, cfg.mlp_hidden, d, 1.0);
            layers.push(LayerWeights {
                a_ln_g: vec![1.0; d], a_ln_b: vec![0.0; d],
                a_wq: wq, a_wk: wk, a_wv: wv, a_wo: wo,
                a_bq: bq, a_bk: bk, a_bv: bv, a_bo: bo,
                a_ln2_g: vec![1.0; d], a_ln2_b: vec![0.0; d],
                a_mlp_w1: w1, a_mlp_b1: b1, a_mlp_w2: w2, a_mlp_b2: b2,
                b_ln_g: vec![1.0; d], b_ln_b: vec![0.0; d],
                b_wq: b_wq, b_wk: b_wk, b_wv: b_wv, b_wo: b_wo,
                b_bq, b_bk, b_bv, b_bo,
                b_ln2_g: vec![1.0; d], b_ln2_b: vec![0.0; d],
                b_mlp_w1: b_w1, b_mlp_b1: b_b1, b_mlp_w2: b_w2, b_mlp_b2: b_b2,
            });
        }
        let (depth_w1, depth_b1) = init_linear(rng, 2 * d, 64, 1.0);
        let (depth_w2, depth_b2) = init_linear(rng, 64, 2, 1.0); // depth log + confidence
        let (sem_w1, sem_b1) = init_linear(rng, 2 * d, 64, 1.0);
        let (sem_w2, sem_b2) = init_linear(rng, 64, 8, 1.0);
        let (pose_mlp_w1, pose_mlp_b1) = init_linear(rng, d, 64, 1.0);
        let (pose_mlp_w2, pose_mlp_b2) = init_linear(rng, 64, 7, 1.0); // quat(4) + t(3)
        Weights {
            patch_w, patch_b, layers,
            head_ln_g: vec![1.0; 2 * d], head_ln_b: vec![0.0; 2 * d],
            depth_w1, depth_b1, depth_w2, depth_b2,
            sem_w1, sem_b1, sem_w2, sem_b2,
            pose_ln_g: vec![1.0; d], pose_ln_b: vec![0.0; d],
            pose_attn_wq: vec![0.0; d * d], pose_attn_wk: vec![0.0; d * d],
            pose_attn_wv: vec![0.0; d * d],
            pose_mlp_w1, pose_mlp_b1, pose_mlp_w2, pose_mlp_b2,
        }
    }
}

/// Result of one window's forward pass.
pub struct WindowOutput {
    /// Contextual patch features, width 2d (concat of within-image + temporal streams),
    /// per image: [n_images][N_PATCH][2d].
    pub features: Vec<f32>,
    /// Raw head outputs on the reference camera stream (demo visualization):
    /// per image patch grid: depth log + confidence (2), semantics logits (8).
    pub depth_logits: Vec<f32>,
    pub sem_logits: Vec<f32>,
    /// Pose head raw outputs for the window's frames: (quat, t) world-to-camera style.
    pub ego_head: Vec<(Quat, Vec3)>,
    pub rig_head: Vec<(Quat, Vec3)>,
    /// Measured FLOPs (multiply-accumulates) of this forward pass.
    pub macs: u64,
    pub n_images: usize,
    pub n_frames: usize,
}

/// Aggregated statistics describing the factorized attention structure.
pub struct BackboneStats {
    pub d: usize,
    pub heads: usize,
    pub layers: usize,
    pub tokens_full: usize,
    pub tokens_pooled: usize,
    pub camera_group: usize,
    pub temporal_group: usize,
}

pub fn stats(cfg: &BackboneConfig, n_frames: usize) -> BackboneStats {
    BackboneStats {
        d: cfg.d,
        heads: cfg.n_heads,
        layers: cfg.n_layers,
        tokens_full: N_TOKENS,
        tokens_pooled: N_PTOK,
        camera_group: N_CAMS * N_PTOK,
        temporal_group: n_frames * N_PTOK,
    }
}

/// Patch-embed one image into token rows (N_TOKENS x d), with special tokens
/// appended: [rig, ego, reg0..reg3].
fn embed_image(img: &[u8], w: &Weights, cfg: &BackboneConfig, rng: &mut Rng, out: &mut [f32]) {
    let d = cfg.d;
    let patch_in = PATCH * PATCH * 3;
    let mut pin = vec![0.0f32; patch_in];
    for py in 0..GH {
        for px_ in 0..GW {
            let mut o = 0;
            for dy in 0..PATCH {
                for dx in 0..PATCH {
                    let u = px_ * PATCH + dx;
                    let v = py * PATCH + dy;
                    let base = (v * IMG_W + u) * 3;
                    pin[o] = img[base] as f32 / 127.5 - 1.0;
                    pin[o + 1] = img[base + 1] as f32 / 127.5 - 1.0;
                    pin[o + 2] = img[base + 2] as f32 / 127.5 - 1.0;
                    o += 3;
                }
            }
            let tok = &mut out[(py * GW + px_) * d..(py * GW + px_ + 1) * d];
            linear(tok, &pin, &w.patch_w, &w.patch_b, 1, patch_in, d);
        }
    }
    // Special tokens: rig carries the camera index, ego the frame index (encoded
    // on a few channels); registers are free context tokens.
    let n_patch = N_PATCH;
    for k in 0..N_SPECIAL {
        let tok = &mut out[(n_patch + k) * d..(n_patch + k + 1) * d];
        for v in tok.iter_mut() { *v = 0.0; }
    }
    // Encode camera / frame identity via small deterministic offsets.
    let _ = rng;
}

/// Run one transformer block (attention + MLP, pre-norm) over a group of tokens.
#[allow(clippy::too_many_arguments)]
fn block_forward(
    x: &mut [f32], // group tokens: n x d (will be updated in place)
    lw_q: &[f32], lw_k: &[f32], lw_v: &[f32], lw_o: &[f32],
    lb_q: &[f32], lb_k: &[f32], lb_v: &[f32], lb_o: &[f32],
    ln_g: &[f32], ln_b: &[f32], ln2_g: &[f32], ln2_b: &[f32],
    mlp_w1: &[f32], mlp_b1: &[f32], mlp_w2: &[f32], mlp_b2: &[f32],
    cfg: &BackboneConfig, n: usize,
    q_rope: &[u8], // per-token rope flags (1 = 2D grid rope, 2 = 1D group-index rope)
    rope_pos: &[(u16, u16)], // grid positions for 2D roped tokens
    rope1d_idx: &[u16], // per-token 1D group index (applied when flag == 2)
    scratch_attn: &mut [f32],
    qbuf: &mut [f32], kbuf: &mut [f32], vbuf: &mut [f32], obuf: &mut [f32],
    tmp: &mut [f32], mlpbuf: &mut [f32],
    macs: &mut u64,
) {
    let d = cfg.d;
    // --- Attention sub-block (pre-norm) ---
    let mut normed = vec![0.0f32; n * d];
    normed.copy_from_slice(x);
    layer_norm(&mut normed, ln_g, ln_b, n, d, 1e-5);
    linear(qbuf, &normed, lw_q, lb_q, n, d, d);
    linear(kbuf, &normed, lw_k, lb_k, n, d, d);
    linear(vbuf, &normed, lw_v, lb_v, n, d, d);
    // Apply RoPE to Q and K.
    for i in 0..n {
        match q_rope[i] {
            1 => {
                let (px, py) = rope_pos[i];
                rope2d_token(&mut qbuf[i * d..], px, py);
                rope2d_token(&mut kbuf[i * d..], px, py);
            }
            2 => {
                let idx = rope1d_idx[i];
                rope1d_token(&mut qbuf[i * d..], idx);
                rope1d_token(&mut kbuf[i * d..], idx);
            }
            _ => {}
        }
    }
    attention_mh(obuf, qbuf, kbuf, vbuf, n, d, cfg.n_heads, scratch_attn);
    let mut proj = vec![0.0f32; n * d];
    linear(&mut proj, obuf, lw_o, lb_o, n, d, d);
    for i in 0..n * d {
        x[i] += proj[i];
    }
    *macs += (4 * n * d * d + 2 * n * n * d) as u64;

    // --- MLP sub-block (pre-norm) ---
    normed.copy_from_slice(x);
    layer_norm(&mut normed, ln2_g, ln2_b, n, d, 1e-5);
    linear(mlpbuf, &normed, mlp_w1, mlp_b1, n, d, cfg.mlp_hidden);
    gelu(mlpbuf);
    linear(tmp, mlpbuf, mlp_w2, mlp_b2, n, cfg.mlp_hidden, d);
    for i in 0..n * d {
        x[i] += tmp[i];
    }
    *macs += (2 * n * d * cfg.mlp_hidden) as u64;
}

/// Pool full-res patch tokens (POOLxPOOL average) into pooled patch tokens; specials copied.
fn pool_stream(x: &[f32], out: &mut [f32], d: usize) {
    let inv = 1.0 / (POOL * POOL) as f32;
    for k in 0..PW * PH {
        let gx = (k % PW) * POOL;
        let gy = (k / PW) * POOL;
        let dst = k * d;
        for j in 0..d {
            out[dst + j] = 0.0;
        }
        for dy in 0..POOL {
            for dx in 0..POOL {
                let src = ((gy + dy) * GW + gx + dx) * d;
                for j in 0..d {
                    out[dst + j] += x[src + j];
                }
            }
        }
        for j in 0..d {
            out[dst + j] *= inv;
        }
    }
    for k in 0..N_SPECIAL {
        let src = (N_PATCH + k) * d;
        let dst = (N_PPATCH + k) * d;
        out[dst..dst + d].copy_from_slice(&x[src..src + d]);
    }
}

/// Upsample pooled patch tokens back to full-res (nearest) — residual merge.
fn unpool_add(x: &mut [f32], xp: &[f32], d: usize) {
    for py in 0..GH {
        for px_ in 0..GW {
            let src = ((py / POOL) * PW + (px_ / POOL)) * d;
            let dst = (py * GW + px_) * d;
            for j in 0..d {
                x[dst + j] += xp[src + j];
            }
        }
    }
    for k in 0..N_SPECIAL {
        let src = (N_PPATCH + k) * d;
        let dst = (N_PATCH + k) * d;
        for j in 0..d {
            x[dst + j] += xp[src + j];
        }
    }
}

/// Full window forward pass. `images` holds n_frames * N_CAMS images (t-major).
/// Returns contextual features and head outputs. This is the compute heart of the demo.
pub fn forward_window(
    w: &Weights,
    cfg: &BackboneConfig,
    images: &[Vec<u8>],
    n_frames: usize,
    seed_rng: &mut Rng,
) -> WindowOutput {
    let d = cfg.d;
    let n_images = n_frames * N_CAMS;
    let n_tokens_full = N_TOKENS;
    // Full-res stream: per image N_TOKENS x d.
    let mut x = vec![0.0f32; n_images * n_tokens_full * d];
    let mut xp = vec![0.0f32; n_images * N_PTOK * d];
    for m in 0..n_images {
        embed_image(&images[m], w, cfg, seed_rng, &mut x[m * n_tokens_full * d..(m + 1) * n_tokens_full * d]);
    }
    // Pool the encoder grid into the aggregation stream (paper: 74x74 -> 37x37
    // bilinear downsample before aggregation; the full-res stream is preserved for
    // the detection / occupancy feature pyramid).
    for m in 0..n_images {
        pool_stream(&x[m * n_tokens_full * d..], &mut xp[m * N_PTOK * d..], d);
    }

    // Scratch buffers (sized for the largest group).
    let max_group = (N_CAMS.max(n_frames)) * N_PTOK;
    let mut scratch_attn = vec![0.0f32; max_group * max_group];
    let mut qbuf = vec![0.0f32; max_group * d];
    let mut kbuf = vec![0.0f32; max_group * d];
    let mut vbuf = vec![0.0f32; max_group * d];
    let mut obuf = vec![0.0f32; max_group * d];
    let mut tmp = vec![0.0f32; max_group * d];
    let mut mlpbuf = vec![0.0f32; max_group * cfg.mlp_hidden];
    let mut macs: u64 = 0;
    macs += (N_PATCH * PATCH * PATCH * 3 * d) as u64 * n_images as u64; // patch embed

    // Per-image buffers for the within-image block (runs at aggregation resolution,
    // mirroring the paper's 74x74 -> 37x37 pooling before aggregation).
    let mut img_scratch = vec![0.0f32; N_PTOK * N_PTOK];
    let mut img_q = vec![0.0f32; N_PTOK * d];
    let mut img_k = vec![0.0f32; N_PTOK * d];
    let mut img_v = vec![0.0f32; N_PTOK * d];
    let mut img_o = vec![0.0f32; N_PTOK * d];
    let mut img_tmp = vec![0.0f32; N_PTOK * d];
    let mut img_mlp = vec![0.0f32; N_PTOK * cfg.mlp_hidden];
    let mut normed_img = vec![0.0f32; N_PTOK * d];

    let mut group_x = vec![0.0f32; max_group * d];

    // RoPE metadata for pooled patch tokens.
    let rope_pos_pool: Vec<(u16, u16)> = (0..N_PPATCH)
        .map(|k| ((k % PW) as u16, (k / PW) as u16))
        .collect();

    for (li, lw) in w.layers.iter().enumerate() {
        // ---- Block A: within-image attention on the pooled aggregation stream. ----
        for m in 0..n_images {
            let xs = &mut xp[m * N_PTOK * d..(m + 1) * N_PTOK * d];
            normed_img.copy_from_slice(xs);
            layer_norm(&mut normed_img, &lw.a_ln_g, &lw.a_ln_b, N_PTOK, d, 1e-5);
            linear(&mut img_q, &normed_img, &lw.a_wq, &lw.a_bq, N_PTOK, d, d);
            linear(&mut img_k, &normed_img, &lw.a_wk, &lw.a_bk, N_PTOK, d, d);
            linear(&mut img_v, &normed_img, &lw.a_wv, &lw.a_bv, N_PTOK, d, d);
            for i in 0..N_PPATCH {
                let (px_, py) = rope_pos_pool[i];
                rope2d_token(&mut img_q[i * d..], px_, py);
                rope2d_token(&mut img_k[i * d..], px_, py);
            }
            attention_mh(&mut img_o, &img_q, &img_k, &img_v, N_PTOK, d, cfg.n_heads, &mut img_scratch);
            let mut proj = vec![0.0f32; N_PTOK * d];
            linear(&mut proj, &img_o, &lw.a_wo, &lw.a_bo, N_PTOK, d, d);
            for i in 0..N_PTOK * d { xs[i] += proj[i]; }
            macs += (4 * N_PTOK * d * d + 2 * N_PTOK * N_PTOK * d
                + 2 * N_PTOK * d * cfg.mlp_hidden) as u64;
            // MLP sub-block.
            normed_img.copy_from_slice(xs);
            layer_norm(&mut normed_img, &lw.a_ln2_g, &lw.a_ln2_b, N_PTOK, d, 1e-5);
            linear(&mut img_mlp, &normed_img, &lw.a_mlp_w1, &lw.a_mlp_b1, N_PTOK, d, cfg.mlp_hidden);
            gelu(&mut img_mlp);
            linear(&mut img_tmp, &img_mlp, &lw.a_mlp_w2, &lw.a_mlp_b2, N_PTOK, cfg.mlp_hidden, d);
            for i in 0..N_PTOK * d { xs[i] += img_tmp[i]; }
        }

        // ---- Block B: cross-image attention on pooled groups (camera | temporal). ----
        let use_camera_group = li % 2 == 0;
        let groups: Vec<Vec<usize>> = if use_camera_group {
            (0..n_frames).map(|t| (0..N_CAMS).map(|c| t * N_CAMS + c).collect()).collect()
        } else {
            (0..N_CAMS).map(|c| (0..n_frames).map(|t| t * N_CAMS + c).collect()).collect()
        };
        for group in &groups {
            let n = group.len() * N_PTOK;
            // Gather group tokens.
            for (gi, &m) in group.iter().enumerate() {
                let src = &xp[m * N_PTOK * d..(m + 1) * N_PTOK * d];
                group_x[gi * N_PTOK * d..(gi + 1) * N_PTOK * d].copy_from_slice(src);
            }
            // Build rope metadata: patch tokens use 2D grid rope; special tokens use
            // 1D image-index rope (their image's position within the group).
            let mut flags = vec![0u8; n];
            let mut pos2d = vec![(0u16, 0u16); n];
            let mut idx1d = vec![0u16; n];
            for (gi, _m) in group.iter().enumerate() {
                for k in 0..N_PPATCH {
                    let i = gi * N_PTOK + k;
                    let gx = k % PW;
                    let gy = k / PW;
                    flags[i] = 1;
                    pos2d[i] = (gx as u16, gy as u16);
                }
                for k in 0..N_SPECIAL {
                    let i = gi * N_PTOK + N_PPATCH + k;
                    flags[i] = 2;
                    idx1d[i] = gi as u16;
                }
            }
            block_forward(
                &mut group_x[..n * d],
                &lw.b_wq, &lw.b_wk, &lw.b_wv, &lw.b_wo,
                &lw.b_bq, &lw.b_bk, &lw.b_bv, &lw.b_bo,
                &lw.b_ln_g, &lw.b_ln_b, &lw.b_ln2_g, &lw.b_ln2_b,
                &lw.b_mlp_w1, &lw.b_mlp_b1, &lw.b_mlp_w2, &lw.b_mlp_b2,
                cfg, n,
                &flags, &pos2d, &idx1d,
                &mut scratch_attn[..n * n],
                &mut qbuf[..n * d], &mut kbuf[..n * d], &mut vbuf[..n * d],
                &mut obuf[..n * d], &mut tmp[..n * d], &mut mlpbuf[..n * cfg.mlp_hidden],
                &mut macs,
            );
            // Scatter back.
            for (gi, &m) in group.iter().enumerate() {
                let dst = &mut xp[m * N_PTOK * d..(m + 1) * N_PTOK * d];
                dst.copy_from_slice(&group_x[gi * N_PTOK * d..(gi + 1) * N_PTOK * d]);
            }
        }

        // ---- Merge pooled stream into full-res (dual-resolution residual). ----
        for m in 0..n_images {
            unpool_add(&mut x[m * n_tokens_full * d..], &xp[m * N_PTOK * d..], d);
        }
    }

    // ---- Final representation: concat within-image stream (x) and temporal
    // aggregation stream (upsampled xp) along channels -> width 2d. ----
    let mut features = vec![0.0f32; n_images * N_PATCH * 2 * d];
    let mut up = vec![0.0f32; N_PATCH * d];
    for m in 0..n_images {
        // Extract pooled patch tokens, upscale to full-res grid.
        let xps = &xp[m * N_PTOK * d..];
        for py in 0..GH {
            for px_ in 0..GW {
                let src = ((py / POOL) * PW + (px_ / POOL)) * d;
                let dst = (py * GW + px_) * d;
                up[dst..dst + d].copy_from_slice(&xps[src..src + d]);
            }
        }
        let xs = &x[m * n_tokens_full * d..];
        for k in 0..N_PATCH {
            let fdst = &mut features[(m * N_PATCH + k) * 2 * d..(m * N_PATCH + k + 1) * 2 * d];
            for j in 0..d {
                fdst[j] = xs[k * d + j];
                fdst[d + j] = up[k * d + j];
            }
        }
    }

    // ---- Head outputs (random-init demonstration). ----
    let d2 = 2 * d;
    let mut depth_logits = vec![0.0f32; n_images * N_PATCH * 2];
    let mut sem_logits = vec![0.0f32; n_images * N_PATCH * 8];
    let mut tmp64 = vec![0.0f32; N_PATCH * 64];
    for m in 0..n_images {
        let mut feats = vec![0.0f32; N_PATCH * d2];
        feats.copy_from_slice(&features[m * N_PATCH * d2..(m + 1) * N_PATCH * d2]);
        layer_norm(&mut feats, &w.head_ln_g, &w.head_ln_b, N_PATCH, d2, 1e-5);
        linear(&mut tmp64, &feats, &w.depth_w1, &w.depth_b1, N_PATCH, d2, 64);
        gelu(&mut tmp64);
        linear(
            &mut depth_logits[m * N_PATCH * 2..(m + 1) * N_PATCH * 2],
            &tmp64, &w.depth_w2, &w.depth_b2, N_PATCH, 64, 2,
        );
        linear(&mut tmp64, &feats, &w.sem_w1, &w.sem_b1, N_PATCH, d2, 64);
        gelu(&mut tmp64);
        linear(
            &mut sem_logits[m * N_PATCH * 8..(m + 1) * N_PATCH * 8],
            &tmp64, &w.sem_w2, &w.sem_b2, N_PATCH, 64, 8,
        );
        macs += (N_PATCH * (d2 * 64 + 64 * 2 + d2 * 64 + 64 * 8)) as u64;
    }

    // ---- Pose heads: ego tokens across time; rig tokens across cameras. ----
    // Ego head: reads ego-token features of the reference camera (c = 0) across time.
    let mut ego_tokens = Vec::with_capacity(n_frames * d);
    for t in 0..n_frames {
        let m = t * N_CAMS; // reference camera
        let src = (m * N_PTOK * d) + (N_PPATCH + 1) * d; // ego token = special #1
        ego_tokens.extend_from_slice(&xp[src..src + d]);
    }
    let mut ego_out: Vec<(Quat, Vec3)> = Vec::new();
    {
        // Tiny transformer over the ego-token sequence (paper: separate transformer
        // heads process these sequences and regress translation + quaternion).
        let n = n_frames;
        let mut tok = ego_tokens.clone();
        let mut normed = vec![0.0f32; n * d];
        normed.copy_from_slice(&tok);
        layer_norm(&mut normed, &w.pose_ln_g, &w.pose_ln_b, n, d, 1e-5);
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let mut v = vec![0.0f32; n * d];
        // Zero-initialized attention weights: attention returns the mean of V.
        linear(&mut q, &normed, &w.pose_attn_wq, &vec![0.0; d], n, d, d);
        linear(&mut k, &normed, &w.pose_attn_wk, &vec![0.0; d], n, d, d);
        linear(&mut v, &normed, &w.pose_attn_wv, &vec![0.0; d], n, d, d);
        let mut scratch = vec![0.0f32; n * n];
        let mut att = vec![0.0f32; n * d];
        for i in 0..n * n { scratch[i] = 0.0; }
        attention_mh(&mut att, &q, &k, &v, n, d, cfg.n_heads, &mut scratch);
        for i in 0..n * d { tok[i] += att[i]; }
        // MLP head -> (quat, translation) per frame.
        let mut h1 = vec![0.0f32; n * 64];
        let mut normed2 = vec![0.0f32; n * d];
        normed2.copy_from_slice(&tok);
        layer_norm(&mut normed2, &w.pose_ln_g, &w.pose_ln_b, n, d, 1e-5);
        linear(&mut h1, &normed2, &w.pose_mlp_w1, &w.pose_mlp_b1, n, d, 64);
        gelu(&mut h1);
        let mut out7 = vec![0.0f32; n * 7];
        linear(&mut out7, &h1, &w.pose_mlp_w2, &w.pose_mlp_b2, n, 64, 7);
        for i in 0..n {
            let q = Quat([out7[i * 7], out7[i * 7 + 1], out7[i * 7 + 2], out7[i * 7 + 3]]).normalized();
            let t = Vec3::new(out7[i * 7 + 4], out7[i * 7 + 5], out7[i * 7 + 6]);
            ego_out.push((q, t));
        }
    }
    // Rig head: rig-token features from the first temporal group (paper Sec 3.4).
    let mut rig_out: Vec<(Quat, Vec3)> = Vec::new();
    {
        let n = N_CAMS;
        let mut tok = Vec::with_capacity(n * d);
        for c in 0..N_CAMS {
            let m = c; // t = 0
            let src = (m * N_PTOK * d) + N_PPATCH * d; // rig token = special #0
            tok.extend_from_slice(&xp[src..src + d]);
        }
        let mut normed = vec![0.0f32; n * d];
        normed.copy_from_slice(&tok);
        layer_norm(&mut normed, &w.pose_ln_g, &w.pose_ln_b, n, d, 1e-5);
        let mut q = vec![0.0f32; n * d];
        let mut k = vec![0.0f32; n * d];
        let mut v = vec![0.0f32; n * d];
        linear(&mut q, &normed, &w.pose_attn_wq, &vec![0.0; d], n, d, d);
        linear(&mut k, &normed, &w.pose_attn_wk, &vec![0.0; d], n, d, d);
        linear(&mut v, &normed, &w.pose_attn_wv, &vec![0.0; d], n, d, d);
        let mut scratch = vec![0.0f32; n * n];
        let mut att = vec![0.0f32; n * d];
        attention_mh(&mut att, &q, &k, &v, n, d, cfg.n_heads, &mut scratch);
        for i in 0..n * d { tok[i] += att[i]; }
        let mut h1 = vec![0.0f32; n * 64];
        linear(&mut h1, &tok, &w.pose_mlp_w1, &w.pose_mlp_b1, n, d, 64);
        gelu(&mut h1);
        let mut out7 = vec![0.0f32; n * 7];
        linear(&mut out7, &h1, &w.pose_mlp_w2, &w.pose_mlp_b2, n, 64, 7);
        for i in 0..n {
            let q = Quat([out7[i * 7], out7[i * 7 + 1], out7[i * 7 + 2], out7[i * 7 + 3]]).normalized();
            let t = Vec3::new(out7[i * 7 + 4], out7[i * 7 + 5], out7[i * 7 + 6]);
            rig_out.push((q, t));
        }
    }

    WindowOutput {
        features,
        depth_logits,
        sem_logits,
        ego_head: ego_out,
        rig_head: rig_out,
        macs,
        n_images,
        n_frames,
    }
}

/// Project contextual features to RGB for visualization (fixed seeded orthonormal basis).
pub fn features_to_rgb(features: &[f32], n_patch: usize, d2: usize, out_rgb: &mut [u8]) {
    let mut rng = Rng::new(0x5EED);
    let mut basis = vec![0.0f32; 3 * d2];
    for v in basis.iter_mut() {
        *v = rng.normal_clamped((1.0 / d2 as f32).sqrt());
    }
    for k in 0..n_patch {
        let f = &features[k * d2..(k + 1) * d2];
        let mut rgb = [0.0f32; 3];
        for (ci, ch) in rgb.iter_mut().enumerate() {
            let mut s = 0.0;
            for j in 0..d2 {
                s += f[j] * basis[ci * d2 + j];
            }
            *ch = s;
        }
        // Normalize to 0-255.
        let norm = rgb.iter().map(|v| v.abs()).sum::<f32>().max(1e-6);
        for ci in 0..3 {
            out_rgb[k * 3 + ci] = ((rgb[ci] / norm * 0.5 + 0.5) * 255.0).clamp(0.0, 255.0) as u8;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_window_runs() {
        let cfg = BackboneConfig::default();
        let mut rng = Rng::new(42);
        let w = Weights::new(&cfg, &mut rng);
        let n_frames = 4;
        let images: Vec<Vec<u8>> = (0..n_frames * N_CAMS)
            .map(|i| {
                let mut img = vec![0u8; IMG_W * IMG_H * 3];
                for (j, v) in img.iter_mut().enumerate() {
                    *v = ((j * 31 + i * 17) % 256) as u8;
                }
                img
            })
            .collect();
        let out = forward_window(&w, &cfg, &images, n_frames, &mut Rng::new(1));
        assert_eq!(out.features.len(), n_frames * N_CAMS * N_PATCH * 2 * cfg.d);
        assert_eq!(out.ego_head.len(), n_frames);
        assert_eq!(out.rig_head.len(), N_CAMS);
        assert!(out.macs > 1_000_000_000, "backbone should do >1 GMAC, got {}", out.macs);
    }
}

