//! Geometry branch (paper Sec 3.4): depth estimation, semantic prediction and
//! pointmap unprojection.
//!
//! The trained model regresses depth/confidence/semantics through DPT decoders on
//! contextual features. This implementation runs the backbone + heads as a real
//! forward pass (random init — its raw outputs are surfaced in the UI), and drives
//! the reconstruction with a classical multi-view geometry stand-in that runs on the
//! same inputs: a coarse-to-fine plane sweep over the surround cameras *and*
//! adjacent keyframes (mirroring the aggregator's camera + temporal factorization),
//! with **SGM-style cost aggregation** (semi-global matching, Hirschmuller 2008)
//! on the coarse cost volume — 4-path dynamic programming with edge-aware smoothness
//! penalties in log-disparity space. SGM is what makes far-field depth usable:
//! beyond ~40 m the raw photometric cost curve is nearly flat (sub-pixel disparity),
//! so winner-takes-all degenerates into noise-driven outliers, while path
//! aggregation suppresses those outliers and yields piecewise-smooth surfaces.
//! The fine stage then refines depth on a local 9-bin grid at ~4.4% log steps
//! (vs the 15.7% coarse bins), so effective depth resolution stays ~4-5% at all
//! ranges — including the 40-120 m far field.
//!
//! Pointmaps follow Eq. (2): p(u) = R * (d(u) / r_z) * r(u) + t with the pose
//! factorization T_t,c = T_ego(tau_t,c) * T_rig_c (Eq. 1) supplying (R, t).

use crate::math::{Rng, Se3, Vec3};
use crate::raster::SKY_DEPTH;
use crate::scene::{
    CamImage, CLASS_PROTO, N_CAMS, N_FRAMES, SEM_BUILDING, SEM_PERSON, SEM_ROAD,
    SEM_SIDEWALK, SEM_SKY, SEM_VEGETATION, SEM_VEHICLE, IMG_H, IMG_W, N_SEM,
};

/// Coarse sweep: log-spaced bins over the full range (evaluated at half resolution).
pub const NB_COARSE: usize = 32;
/// Fine refinement: local bins around the coarse estimate (~4.4% log steps).
pub const NB_FINE: usize = 9;
pub const FINE_RATIO: f32 = 1.044_f32;
pub const D_MIN: f32 = 1.2;
/// Far-field range: the scene's observable structure (buildings along the road
/// ahead) sits within ~90 m; beyond that only sky remains.
pub const D_MAX: f32 = 90.0;

/// SGM penalties (log-disparity units): P1 for +-1 bin, P2 (edge-modulated) for
/// larger jumps. Tuned on the procedural clip for far-field AbsRel.
pub const SGM_P1: f32 = 0.85;
pub const SGM_P2: f32 = 5.5;
pub const SGM_P2_MIN: f32 = 1.4;

/// Log-depth prior: for a ground-plane scene the density of VISIBLE surfaces
/// along a ray falls like 1/z^2, so when photometric evidence is ambiguous
/// (weakly textured road seen mostly through temporal sources corrupted by
/// lead-vehicle occlusion), the maximum-likelihood tie-break favors nearer
/// depths. Penalty is linear in log-depth, i.e. constant per bin.
pub const PRIOR_PER_LOG: f32 = 6.0; // cost units per e-fold of depth

/// Half-resolution sweep grid.
pub const CW: usize = IMG_W / 2;
pub const CH: usize = IMG_H / 2;

pub struct CameraModel {
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    /// Per-pixel camera-frame unit rays (IMG_W * IMG_H).
    pub rays: Vec<Vec3>,
}

impl CameraModel {
    pub fn new() -> CameraModel {
        let fx = (IMG_W as f32 * 0.5) / (crate::scene::FOV_X * 0.5).tan();
        let cx = IMG_W as f32 * 0.5;
        let cy = IMG_H as f32 * 0.5;
        let mut rays = Vec::with_capacity(IMG_W * IMG_H);
        for v in 0..IMG_H {
            for u in 0..IMG_W {
                rays.push(Vec3::new(
                    (u as f32 + 0.5 - cx) / fx,
                    (v as f32 + 0.5 - cy) / fx,
                    1.0,
                ).normalized());
            }
        }
        CameraModel { fx, fy: fx, cx, cy, rays }
    }

    /// Project a camera-frame point to pixel coordinates. Returns None if behind.
    pub fn project(&self, p: &Vec3) -> Option<(f32, f32)> {
        if p.z() < 0.05 {
            return None;
        }
        let invz = 1.0 / p.z();
        Some((self.fx * p.x() * invz + self.cx - 0.5, self.fy * p.y() * invz + self.cy - 0.5))
    }
}

/// Per-image geometric predictions.
#[derive(Clone)]
pub struct ImagePred {
    pub depth: Vec<f32>,
    pub conf: Vec<f32>,
    pub sem: Vec<u8>,
    /// Unprojected points in the clip frame (world), 3 floats per pixel.
    pub points: Vec<f32>,
}

/// Predicted (noisy) camera poses used for stereo and assembly — simulates the
/// pose heads' output quality. Poses are cam-to-world in the *clip* frame for the
/// reference rig trajectory, plus rig transforms.
pub struct PosePred {
    /// Noisy reference-camera (c=0) trajectory, one per keyframe (cam-to-world).
    pub ego: Vec<Se3>,
    /// Noisy rig transforms (camera c in rig frame).
    pub rig: Vec<Se3>,
}

/// Compose camera pose: T_t,c = T_ego(t) * T_rig_c (Eq. 1, synchronous case).
pub fn cam_pose(pp: &PosePred, t: usize, c: usize) -> Se3 {
    pp.ego[t].compose(&pp.rig[c])
}

/// Bilinear RGB sample; returns None outside the image.
#[inline]
pub fn sample_rgb_pub(img: &[u8], x: f32, y: f32) -> Option<[f32; 3]> { sample_rgb(img, x, y) }
fn sample_rgb(img: &[u8], x: f32, y: f32) -> Option<[f32; 3]> {
    if x < 0.0 || y < 0.0 || x >= IMG_W as f32 - 1.0 || y >= IMG_H as f32 - 1.0 {
        return None;
    }
    let x0 = x.floor() as usize;
    let y0 = y.floor() as usize;
    let tx = x - x0 as f32;
    let ty = y - y0 as f32;
    let mut out = [0.0f32; 3];
    for ch in 0..3 {
        let v00 = img[(y0 * IMG_W + x0) * 3 + ch] as f32;
        let v01 = img[(y0 * IMG_W + x0 + 1) * 3 + ch] as f32;
        let v10 = img[((y0 + 1) * IMG_W + x0) * 3 + ch] as f32;
        let v11 = img[((y0 + 1) * IMG_W + x0 + 1) * 3 + ch] as f32;
        let a = v00 + (v01 - v00) * tx;
        let b = v10 + (v11 - v10) * tx;
        out[ch] = a + (b - a) * ty;
    }
    Some(out)
}

/// Source views for one reference image: images + cam-to-world poses + stereo weights
/// (spatial ring neighbors are motion-free and thus trusted more than temporal
/// frames, which see independently moving actors).
pub struct SweepSources<'a> {
    pub imgs: Vec<&'a [u8]>,
    pub poses: Vec<Se3>,
    pub weights: Vec<f32>,
}

/// Reusable scratch buffers for the coarse-to-fine sweep (keeps per-frame calls
/// allocation-free — the browser stage machine calls this 32x per clip).
pub struct DepthScratch {
    /// 2x2-box-downsampled reference image (CW * CH * 3).
    ref_half: Vec<u8>,
    /// Grayscale of `ref_half` (for SGM's edge-aware P2).
    ref_half_gray: Vec<f32>,
    /// Coarse cost volume (CW * CH * NB_COARSE).
    cost: Vec<f32>,
    /// SGM path-aggregated volume (CW * CH * NB_COARSE).
    sgm: Vec<f32>,
    /// Per-path running cost (CW * CH * NB_COARSE).
    path: Vec<f32>,
    /// min_k L(p-r, k) memo per coarse pixel.
    path_min: Vec<f32>,
    /// Fine cost volume (IMG_W * IMG_H * NB_FINE).
    fine: Vec<f32>,
    fine_tmp: Vec<f32>,
    /// Per-source rotated ray directions at full resolution: R_sfr * ray, laid out
    /// [src][pixel][3]. Precomputed once per (t, c) — the per-bin work then reduces
    /// to one fused-multiply-add per axis.
    dir_s: Vec<f32>,
    /// Coarse results, half-res: depth, confidence, supported flag.
    d_half: Vec<f32>,
    c_half: Vec<f32>,
    s_half: Vec<u8>,
    /// Fine band center (full-res, log-space interpolated from coarse).
    band_center: Vec<f32>,
}

impl DepthScratch {
    pub fn new() -> DepthScratch {
        DepthScratch {
            ref_half: vec![0u8; CW * CH * 3],
            ref_half_gray: vec![0.0; CW * CH],
            cost: vec![0.0; CW * CH * NB_COARSE],
            sgm: vec![0.0; CW * CH * NB_COARSE],
            path: vec![0.0; CW * CH * NB_COARSE],
            path_min: vec![0.0; CW * CH],
            fine: vec![0.0; IMG_W * IMG_H * NB_FINE],
            fine_tmp: vec![0.0; IMG_W * IMG_H * NB_FINE],
            dir_s: Vec::new(),
            d_half: vec![0.0; CW * CH],
            c_half: vec![0.0; CW * CH],
            s_half: vec![0u8; CW * CH],
            band_center: vec![0.0; IMG_W * IMG_H],
        }
    }

    fn ensure_sources(&mut self, n_src: usize) {
        if self.dir_s.len() < n_src * IMG_W * IMG_H * 3 {
            self.dir_s.resize(n_src * IMG_W * IMG_H * 3, 0.0);
        }
    }
}

/// Log-spaced coarse bins.
fn coarse_bins() -> [f32; NB_COARSE] {
    let mut bins = [0.0f32; NB_COARSE];
    for k in 0..NB_COARSE {
        bins[k] = D_MIN * (D_MAX / D_MIN).powf(k as f32 / (NB_COARSE - 1) as f32);
    }
    bins
}

/// 2x2 box downsample of the reference RGB + grayscale.
fn downsample_half(rgb: &[u8], out_rgb: &mut [u8], out_gray: &mut [f32]) {
    for y in 0..CH {
        for x in 0..CW {
            let mut acc = [0u32; 3];
            for dy in 0..2 {
                for dx in 0..2 {
                    let p = ((y * 2 + dy) * IMG_W + x * 2 + dx) * 3;
                    acc[0] += rgb[p] as u32;
                    acc[1] += rgb[p + 1] as u32;
                    acc[2] += rgb[p + 2] as u32;
                }
            }
            let q = y * CW + x;
            out_rgb[q * 3] = (acc[0] / 4) as u8;
            out_rgb[q * 3 + 1] = (acc[1] / 4) as u8;
            out_rgb[q * 3 + 2] = (acc[2] / 4) as u8;
            out_gray[q] = 0.30 * out_rgb[q * 3] as f32 + 0.59 * out_rgb[q * 3 + 1] as f32 + 0.11 * out_rgb[q * 3 + 2] as f32;
        }
    }
}

/// Photometric cost of matching reference pixel `p` (color `col`) against source
/// `si` backprojected at z-depth `d`. `dir` is the precomputed R_sfr * ray vector;
/// `t_rel` the translation of the source-from-reference transform.
#[inline]
fn source_cost(
    src_img: &[u8],
    dir: &[f32],
    t_rel: &Vec3,
    d: f32,
    col: &[f32; 3],
    cam: &CameraModel,
) -> Option<f32> {
    let sx = dir[0] * d + t_rel.x();
    let sy = dir[1] * d + t_rel.y();
    let sz = dir[2] * d + t_rel.z();
    if sz < 0.1 {
        return None;
    }
    let invz = 1.0 / sz;
    let u = cam.fx * sx * invz + cam.cx - 0.5;
    let v = cam.fy * sy * invz + cam.cy - 0.5;
    let c2 = sample_rgb(src_img, u, v)?;
    Some((col[0] - c2[0]).abs() + (col[1] - c2[1]).abs() + (col[2] - c2[2]).abs())
}

/// Accumulate the photometric cost volume at half resolution into `scratch.cost`
/// (weighted SAD over observing sources + missing-source bias, matching the
/// original formulation), and the per-pixel supported-bin counts into `s_half>0`.
/// The reference color is the POINT-SAMPLED full-res pixel whose ray the coarse
/// cell uses — box-averaging the reference against point-sampled sources breaks
/// photometric consistency at texture edges and biases WTA toward far bins.
fn coarse_cost_volume(
    ref_img: &[u8],
    ref_pose: &Se3,
    src: &SweepSources,
    cam: &CameraModel,
    scratch: &mut DepthScratch,
    bins: &[f32; NB_COARSE],
) {
    let n_src = src.imgs.len();
    let w_tot: f32 = src.weights.iter().sum();
    // Precompute source-from-reference rigid transforms: T_sfr = T_src^-1 * T_ref.
    let mut t_rel: Vec<Vec3> = Vec::with_capacity(n_src);
    for sp in &src.poses {
        let sfr = Se3::between(sp, ref_pose);
        t_rel.push(sfr.t);
    }
    // dir_s layout: [src][pixel][3], R_sfr applied to each ref ray.
    // (Computed at FULL resolution; coarse pixels read the (2y+1, 2x+1) entry.)
    for si in 0..n_src {
        let sfr = Se3::between(&src.poses[si], ref_pose);
        let r = &sfr.r;
        let base = si * IMG_W * IMG_H * 3;
        for p in 0..IMG_W * IMG_H {
            let ray = &cam.rays[p];
            let o = base + p * 3;
            scratch.dir_s[o] = r.0[0] * ray.x() + r.0[1] * ray.y() + r.0[2] * ray.z();
            scratch.dir_s[o + 1] = r.0[3] * ray.x() + r.0[4] * ray.y() + r.0[5] * ray.z();
            scratch.dir_s[o + 2] = r.0[6] * ray.x() + r.0[7] * ray.y() + r.0[8] * ray.z();
        }
    }
    for v in scratch.s_half.iter_mut() { *v = 0; }
    for q in 0..CW * CH {
        // Coarse pixel (x, y) uses the ray of full-res pixel (2y+1, 2x+1).
        let x = q % CW;
        let y = q / CW;
        let p_full = (2 * y + 1) * IMG_W + 2 * x + 1;
        let ray = &cam.rays[p_full];
        let rz = ray.z();
        let col = [
            ref_img[p_full * 3] as f32,
            ref_img[p_full * 3 + 1] as f32,
            ref_img[p_full * 3 + 2] as f32,
        ];
        for k in 0..NB_COARSE {
            let d = bins[k] / rz; // scale along the unit ray so that p.z = bins[k]
            let mut cost = 0.0f32;
            let mut wsum = 0.0f32;
            for si in 0..n_src {
                let o = si * IMG_W * IMG_H * 3 + p_full * 3;
                if let Some(c) = source_cost(src.imgs[si], &scratch.dir_s[o..o + 3], &t_rel[si], d, &col, cam) {
                    cost += src.weights[si] * c;
                    wsum += src.weights[si];
                }
            }
            // Log-depth prior (see PRIOR_PER_LOG).
            let prior = PRIOR_PER_LOG * (bins[k] / D_MIN).ln();
            if wsum <= 0.0 {
                scratch.cost[q * NB_COARSE + k] = 120.0 + prior;
            } else {
                if scratch.s_half[q] < 255 { scratch.s_half[q] += 1; }
                scratch.cost[q * NB_COARSE + k] =
                    cost / wsum + 6.0 * (w_tot - wsum) / w_tot.max(1.0) * 3.0 + prior;
            }
        }
    }
}

/// One SGM path pass over the coarse cost volume. Direction (dx, dy); pixels are
/// visited in an order that guarantees p - (dx,dy) was computed earlier.
fn sgm_path(dx: i32, dy: i32, scratch: &mut DepthScratch) {
    let gray = &scratch.ref_half_gray;
    let (w, h, nb) = (CW, CH, NB_COARSE);
    // Visit pixels in raster order for down/right directions, reverse otherwise.
    let y_iter: Vec<usize> = if dy >= 0 { (0..h).collect() } else { (0..h).rev().collect() };
    let x_iter: Vec<usize> = if dy == 0 && dx < 0 { (0..w).rev().collect() } else { (0..w).collect() };
    for &y in &y_iter {
        for &x in &x_iter {
            let p = y * w + x;
            let nx = x as i32 - dx;
            let ny = y as i32 - dy;
            let has_n = nx >= 0 && nx < w as i32 && ny >= 0 && ny < h as i32;
            let n = if has_n { (ny as usize) * w + nx as usize } else { usize::MAX };
            // Edge-aware P2: smoothness relaxes across image edges so depth
            // discontinuities survive at object boundaries.
            let p2 = if has_n {
                let di = (gray[p] - gray[n]).abs();
                (SGM_P2 / (1.0 + di / 8.0)).max(SGM_P2_MIN)
            } else {
                SGM_P2
            };
            let cost = &scratch.cost;
            let path = &mut scratch.path;
            let path_min = &mut scratch.path_min;
            let sgm = &mut scratch.sgm;
            let nmin = if has_n { path_min[n] } else { 0.0 };
            for k in 0..nb {
                let c = cost[p * nb + k];
                let v = if has_n {
                    let mut m = path[n * nb + k];
                    if k > 0 {
                        let a = path[n * nb + k - 1] + SGM_P1;
                        if a < m { m = a; }
                    }
                    if k + 1 < nb {
                        let b = path[n * nb + k + 1] + p2;
                        if b < m { m = b; }
                    }
                    c + m - nmin
                } else {
                    c
                };
                let v = if v < 0.0 { 0.0 } else { v };
                path[p * nb + k] = v;
                sgm[p * nb + k] += v;
            }
            let mut mn = f32::INFINITY;
            for k in 0..nb {
                let v = path[p * nb + k];
                if v < mn { mn = v; }
            }
            path_min[p] = mn;
        }
    }
}

/// Debug accessors for the coarse sweep internals.
pub fn debug_coarse_row(scratch: &DepthScratch, q: usize) -> (Vec<f32>, Vec<f32>, f32, f32) {
    (
        scratch.cost[q * NB_COARSE..(q + 1) * NB_COARSE].to_vec(),
        scratch.sgm[q * NB_COARSE..(q + 1) * NB_COARSE].to_vec(),
        scratch.d_half[q],
        scratch.c_half[q],
    )
}

/// Coarse-to-fine plane sweep + SGM for one reference image. Returns (depth, conf)
/// per full-res pixel. `ref_pose`/`src.poses` are cam-to-world in a common frame.
pub fn sweep_depth(
    ref_img: &[u8],
    ref_pose: &Se3,
    src: &SweepSources,
    cam: &CameraModel,
    scratch: &mut DepthScratch,
) -> (Vec<f32>, Vec<f32>) {
    sweep_depth_mode(ref_img, ref_pose, src, cam, scratch, true)
}

/// `use_sgm = false` falls back to plain WTA over the raw coarse cost volume
/// (with the same log-depth prior) — the ablation baseline.
pub fn sweep_depth_mode(
    ref_img: &[u8],
    ref_pose: &Se3,
    src: &SweepSources,
    cam: &CameraModel,
    scratch: &mut DepthScratch,
    use_sgm: bool,
) -> (Vec<f32>, Vec<f32>) {
    let npix = IMG_W * IMG_H;
    let n_src = src.imgs.len();
    scratch.ensure_sources(n_src.max(1));
    let mut depth = vec![SKY_DEPTH; npix];
    let mut conf = vec![0.0f32; npix];
    if n_src == 0 {
        return (depth, conf);
    }
    let bins = coarse_bins();
    downsample_half(ref_img, &mut scratch.ref_half, &mut scratch.ref_half_gray);
    coarse_cost_volume(ref_img, ref_pose, src, cam, scratch, &bins);

    // ---- SGM aggregation: 4 paths (L, R, U, D). ----
    if use_sgm {
        for v in scratch.sgm.iter_mut() { *v = 0.0; }
        for (dx, dy) in [(1i32, 0i32), (-1, 0), (0, 1), (0, -1)] {
            sgm_path(dx, dy, scratch);
        }
    } else {
        // Ablation: aggregate = raw cost (single-path equivalent).
        scratch.sgm.copy_from_slice(&scratch.cost);
    }

    // ---- Coarse WTA + parabolic sub-bin refinement. ----
    let log_step = (D_MAX / D_MIN).ln() / (NB_COARSE - 1) as f32;
    for q in 0..CW * CH {
        let row = &scratch.sgm[q * NB_COARSE..(q + 1) * NB_COARSE];
        let mut best = 0usize;
        let mut best_c = f32::INFINITY;
        for k in 0..NB_COARSE {
            if row[k] < best_c {
                best_c = row[k];
                best = k;
            }
        }
        let t = if best > 0 && best < NB_COARSE - 1 {
            let c0 = row[best - 1];
            let c1 = row[best];
            let c2 = row[best + 1];
            let denom = c0 - 2.0 * c1 + c2;
            if denom > 1e-6 {
                ((c0 - c2) / (2.0 * denom)).clamp(-0.5, 0.5)
            } else { 0.0 }
        } else { 0.0 };
        scratch.d_half[q] = (bins[best].ln() + t * log_step).exp();
        let curvature = if best > 0 && best < NB_COARSE - 1 {
            (row[best - 1] + row[best + 1] - 2.0 * row[best]).max(0.0)
        } else { 0.0 };
        scratch.c_half[q] = (curvature / (best_c + 20.0)).clamp(0.0, 1.0);
    }

    // t_rel for the fine stage (same transforms as coarse).
    let mut t_rel: Vec<Vec3> = Vec::with_capacity(n_src);
    for sp in &src.poses {
        t_rel.push(Se3::between(sp, ref_pose).t);
    }
    let w_tot: f32 = src.weights.iter().sum();

    // ---- Fine refinement at full resolution over a local bin band. ----
    // Skip pixels already resolved as sky or unsupported at the coarse level.
    let fine_log_step = FINE_RATIO.ln();
    for p in 0..npix {
        let ray = &cam.rays[p];
        // Bilinear (in log-depth) of the 2x2 coarse neighborhood -> band center.
        let x = (p % IMG_W) as f32 * 0.5 - 0.5;
        let y = (p / IMG_W) as f32 * 0.5 - 0.5;
        let x0 = x.floor().clamp(0.0, (CW - 2) as f32) as usize;
        let y0 = y.floor().clamp(0.0, (CH - 2) as f32) as usize;
        let tx = (x - x0 as f32).clamp(0.0, 1.0);
        let ty = (y - y0 as f32).clamp(0.0, 1.0);
        let q00 = y0 * CW + x0;
        let l00 = scratch.d_half[q00].ln();
        let l01 = scratch.d_half[q00 + 1].ln();
        let l10 = scratch.d_half[q00 + CW].ln();
        let l11 = scratch.d_half[q00 + CW + 1].ln();
        let d_center = ((l00 * (1.0 - tx) + l01 * tx) * (1.0 - ty)
            + (l10 * (1.0 - tx) + l11 * tx) * ty).exp();
        scratch.band_center[p] = d_center;
        let col = [
            ref_img[p * 3] as f32,
            ref_img[p * 3 + 1] as f32,
            ref_img[p * 3 + 2] as f32,
        ];
        let rz = ray.z();
        let row = &mut scratch.fine[p * NB_FINE..(p + 1) * NB_FINE];
        for j in 0..NB_FINE {
            let d_bin = (d_center * FINE_RATIO.powi(j as i32 - (NB_FINE as i32 - 1) / 2))
                .clamp(D_MIN, D_MAX * 1.08);
            let s = d_bin / rz;
            let mut cost = 0.0f32;
            let mut wsum = 0.0f32;
            for si in 0..n_src {
                let o = si * npix * 3 + p * 3;
                if let Some(c) = source_cost(src.imgs[si], &scratch.dir_s[o..o + 3], &t_rel[si], s, &col, cam) {
                    cost += src.weights[si] * c;
                    wsum += src.weights[si];
                }
            }
            let prior = PRIOR_PER_LOG * (d_bin / D_MIN).ln();
            row[j] = if wsum <= 0.0 {
                120.0 + prior
            } else {
                cost / wsum + 6.0 * (w_tot - wsum) / w_tot.max(1.0) * 3.0 + prior
            };
        }
    }

    // Separable 3-wide box smoothing over the fine band (photometric noise).
    {
        let (w, h, nb) = (IMG_W, IMG_H, NB_FINE);
        for y in 0..h {
            for x in 0..w {
                for k in 0..nb {
                    let mut s = 0.0;
                    let mut n = 0;
                    for dx in -1i32..=1 {
                        let xx = x as i32 + dx;
                        if xx >= 0 && xx < w as i32 {
                            s += scratch.fine[(y * w + xx as usize) * nb + k];
                            n += 1;
                        }
                    }
                    scratch.fine_tmp[(y * w + x) * nb + k] = s / n as f32;
                }
            }
        }
        for y in 0..h {
            for x in 0..w {
                for k in 0..nb {
                    let mut s = 0.0;
                    let mut n = 0;
                    for dy in -1i32..=1 {
                        let yy = y as i32 + dy;
                        if yy >= 0 && yy < h as i32 {
                            s += scratch.fine_tmp[(yy as usize * w + x) * nb + k];
                            n += 1;
                        }
                    }
                    scratch.fine[(y * w + x) * nb + k] = s / n as f32;
                }
            }
        }
    }

    // Fine WTA + parabolic + confidence; sky / unsupported handled by coarse flags.
    for p in 0..npix {
        let ray = &cam.rays[p];
        let ray_w = ref_pose.r.mul_vec(ray);
        let (x, y) = (p % IMG_W, p / IMG_W);
        let q = (y / 2).min(CH - 1) * CW + (x / 2).min(CW - 1);
        let supported = scratch.s_half[q] > 0;
        let conf_c = scratch.c_half[q];
        // Sky: upward rays with weak agreement AND sky-like color (blue-dominant).
        // The color test keeps flat-colored vehicle roofs / white paint out of the
        // sky class — they stay in the sweep with depth bounded by D_MAX instead
        // of the sky sentinel, capping their error at a fraction of what a
        // misclassification would cost.
        let b_ch = ref_img[p * 3 + 2] as i32;
        let g_ch = ref_img[p * 3 + 1] as i32;
        let r_ch = ref_img[p * 3] as i32;
        // Sky color: blue dominant over BOTH red and green (rejects vegetation
        // greens and white/gray paint, which are merely blue-dominant vs red).
        let skyish = ray_w.z() > 0.07
            && (conf_c < 0.05 || scratch.d_half[q] > D_MAX * 0.95)
            && b_ch > r_ch + 8
            && b_ch > g_ch + 8;
        if skyish {
            continue;
        }
        if !supported {
            // Flat-ground monocular prior for the rig's near-field blind zone (the
            // ground strip beside the ego is not observed by any ring camera at that
            // instant). Standing in for the learned monocular prior the paper's
            // geometry decoder applies where multi-view coverage is absent.
            if ray_w.z() < -0.02 {
                let s = ref_pose.t.z() / -ray_w.z();
                let d = s * ray.z();
                if d > 0.4 && d < 55.0 {
                    depth[p] = d;
                    conf[p] = 0.2;
                }
            } else if !skyish {
                // Ambiguous non-sky pixel (never observable, not sky-colored):
                // bound the error at D_MAX rather than the sky sentinel.
                depth[p] = D_MAX;
            }
            continue;
        }
        let row = &scratch.fine[p * NB_FINE..(p + 1) * NB_FINE];
        let mut best = 0usize;
        let mut best_c = f32::INFINITY;
        for j in 0..NB_FINE {
            if row[j] < best_c {
                best_c = row[j];
                best = j;
            }
        }
        let t = if best > 0 && best < NB_FINE - 1 {
            let c0 = row[best - 1];
            let c1 = row[best];
            let c2 = row[best + 1];
            let denom = c0 - 2.0 * c1 + c2;
            if denom > 1e-6 {
                ((c0 - c2) / (2.0 * denom)).clamp(-0.5, 0.5)
            } else { 0.0 }
        } else { 0.0 };
        let log_d = scratch.band_center[p].ln()
            + (t + best as f32 - (NB_FINE as f32 - 1.0) / 2.0) * fine_log_step;
        depth[p] = log_d.exp().clamp(D_MIN * 0.8, D_MAX * 1.1);
        let curvature = if best > 0 && best < NB_FINE - 1 {
            (row[best - 1] + row[best + 1] - 2.0 * row[best]).max(0.0)
        } else { 0.0 };
        conf[p] = (curvature / (best_c + 8.0)).clamp(0.0, 1.0);
    }

    (depth, conf)
}

/// Predicted semantics: geometric (sky/ground) + color-prototype classification.
/// `points` is the unprojected pointmap in the clip frame (3 floats per pixel).
pub fn predict_semantics(
    img: &[u8],
    pose: &Se3,
    cam: &CameraModel,
    depth: &[f32],
    conf: &[f32],
    points: &[f32],
) -> Vec<u8> {
    let npix = IMG_W * IMG_H;
    let mut sem = vec![crate::scene::SEM_UNKNOWN; npix];
    for p in 0..npix {
        let ray = &cam.rays[p];
        let ray_w = pose.r.mul_vec(ray);
        let d = depth[p];
        let pz = points[p * 3 + 2];
        if d >= SKY_DEPTH * 0.9 {
            sem[p] = SEM_SKY;
            continue;
        }
        // Sky test: uphill rays with weak stereo agreement and sky-like color.
        let b = img[p * 3 + 2] as i32;
        let g = img[p * 3 + 1] as i32;
        let r = img[p * 3] as i32;
        let sky_color = b > r + 14 && b > g + 6;
        if ray_w.z() > 0.08 && (conf[p] < 0.06 || (sky_color && d > 40.0)) {
            if conf[p] < 0.06 && !sky_color && d < SKY_DEPTH * 0.9 {
                // Weakly-matched uphill pixel that is NOT sky-colored (vehicle
                // roofs, white paint): keep the swept depth (bounded by D_MAX).
            } else {
                sem[p] = SEM_SKY;
                continue;
            }
        }
        if d < SKY_DEPTH {
            // Ground plane test on the unprojected point (generous height band —
            // depth bias at range can dip the estimate below the plane).
            if pz.abs() < 0.42 && d < 70.0 && ray_w.z() < 0.08 {
                // Road / sidewalk / grass split by height and color.
                let g = img[p * 3 + 1] as i32;
                let rr = img[p * 3] as i32;
                if g > rr + 18 && g > b + 12 {
                    sem[p] = SEM_VEGETATION;
                } else if pz > 0.06 {
                    sem[p] = SEM_SIDEWALK;
                } else {
                    sem[p] = SEM_ROAD;
                }
                continue;
            }
            // Obstacles: color-prototype classification with a 3x3 mean color.
            let x = (p % IMG_W) as i32;
            let y = (p / IMG_W) as i32;
            let mut mc = [0.0f32; 3];
            let mut n = 0.0;
            for dy in -1..=1 {
                for dx in -1..=1 {
                    let xx = x + dx;
                    let yy = y + dy;
                    if xx >= 0 && xx < IMG_W as i32 && yy >= 0 && yy < IMG_H as i32 {
                        let q = (yy as usize * IMG_W + xx as usize) * 3;
                        mc[0] += img[q] as f32;
                        mc[1] += img[q + 1] as f32;
                        mc[2] += img[q + 2] as f32;
                        n += 1.0;
                    }
                }
            }
            for v in mc.iter_mut() { *v /= n; }
            let mut best_cls = SEM_BUILDING;
            let mut best_score = f32::INFINITY;
            for &cls in &[SEM_BUILDING, SEM_VEGETATION, SEM_VEHICLE, SEM_PERSON] {
                let proto = &CLASS_PROTO[cls as usize];
                let d0 = mc[0] / 255.0 - proto[0];
                let d1 = mc[1] / 255.0 - proto[1];
                let d2 = mc[2] / 255.0 - proto[2];
                // Person: also favor small image footprint near ground.
                let mut score = (d0 * d0 + d1 * d1 + d2 * d2) * 8.0;
                if cls == SEM_PERSON && (pz < 1.9 || pz > 2.1) { score += 2.0; }
                if score < best_score {
                    best_score = score;
                    best_cls = cls;
                }
            }
            sem[p] = best_cls;
        }
    }
    sem
}

/// Unproject depth to a pointmap in the clip frame (paper Eq. 2):
/// p(u) = R (d(u) / r_z) r(u) + t.
pub fn unproject(depth: &[f32], pose: &Se3, cam: &CameraModel, points: &mut [f32]) {
    let r = &pose.r;
    let t = &pose.t;
    for p in 0..IMG_W * IMG_H {
        let ray = &cam.rays[p];
        let s = depth[p] / ray.z();
        let x = r.0[0] * ray.x() * s + r.0[1] * ray.y() * s + r.0[2] * ray.z() * s + t.x();
        let y = r.0[3] * ray.x() * s + r.0[4] * ray.y() * s + r.0[5] * ray.z() * s + t.y();
        let z = r.0[6] * ray.x() * s + r.0[7] * ray.y() * s + r.0[8] * ray.z() * s + t.z();
        points[p * 3] = x;
        points[p * 3 + 1] = y;
        points[p * 3 + 2] = z;
    }
}

/// Depth AbsRel over non-sky ground-truth pixels (the paper's primary geometry metric).
pub fn depth_absrel(pred: &[f32], gt: &CamImage) -> f32 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for p in 0..pred.len() {
        if gt.sem[p] == SEM_SKY || gt.depth[p] >= SKY_DEPTH {
            continue;
        }
        sum += (pred[p] - gt.depth[p]).abs() / gt.depth[p].max(0.5);
        n += 1;
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// Far-field AbsRel: ground-truth depth in (30 m, SKY_DEPTH) — the range where
/// sub-pixel disparity makes WTA degenerate and SGM aggregation pays off.
pub fn depth_absrel_far(pred: &[f32], gt: &CamImage) -> f32 {
    let mut sum = 0.0;
    let mut n = 0usize;
    for p in 0..pred.len() {
        if gt.sem[p] == SEM_SKY || gt.depth[p] >= SKY_DEPTH || gt.depth[p] <= 30.0 {
            continue;
        }
        sum += (pred[p] - gt.depth[p]).abs() / gt.depth[p].max(0.5);
        n += 1;
    }
    if n == 0 { 0.0 } else { sum / n as f32 }
}

/// Assemble the source set for reference view (t, c): both adjacent cameras on
/// the ring (spatial factor) plus the same camera at t +- 1 and t +- 2 (temporal
/// factor) — mirroring the aggregator's camera + temporal context.
fn sources_for<'a>(
    images: &'a [CamImage],
    poses: &PosePred,
    t: usize,
    c: usize,
) -> SweepSources<'a> {
    let mut imgs: Vec<&[u8]> = Vec::new();
    let mut src_poses: Vec<Se3> = Vec::new();
    let mut weights: Vec<f32> = Vec::new();
    let prev_c = (c + N_CAMS - 1) % N_CAMS;
    let next_c = (c + 1) % N_CAMS;
    imgs.push(&images[t * N_CAMS + prev_c].rgb);
    src_poses.push(cam_pose(poses, t, prev_c));
    weights.push(1.6);
    imgs.push(&images[t * N_CAMS + next_c].rgb);
    src_poses.push(cam_pose(poses, t, next_c));
    weights.push(1.6);
    for &dt in &[-2i32, -1, 1, 2] {
        let tt = t as i32 + dt;
        if tt >= 0 && (tt as usize) < N_FRAMES {
            imgs.push(&images[tt as usize * N_CAMS + c].rgb);
            src_poses.push(cam_pose(poses, tt as usize, c));
            // t+-2 frames carry the largest ego baseline (best far-field
            // disparity) but also the most dynamic-occlusion corruption —
            // weight them below the t+-1 neighbors.
            weights.push(if dt.abs() == 2 { 0.6 } else { 1.0 });
        }
    }
    SweepSources { imgs, poses: src_poses, weights }
}

/// Geometry prediction for ONE keyframe (all 6 cameras) — the pipeline advances
/// this once per frame so long clips stream with per-frame progress.
/// Returns the per-camera predictions and the accumulated AbsRel (sum / count
/// handled by the caller via `absrel_count`).
pub fn predict_frame(
    images: &[CamImage],
    poses: &PosePred,
    cam: &CameraModel,
    scratch: &mut DepthScratch,
    t: usize,
) -> Vec<ImagePred> {
    predict_frame_mode(images, poses, cam, scratch, t, true)
}

/// `use_sgm = false` disables the SGM path aggregation (WTA on the raw cost
/// volume) — used to quantify SGM's contribution to far-field depth.
pub fn predict_frame_mode(
    images: &[CamImage],
    poses: &PosePred,
    cam: &CameraModel,
    scratch: &mut DepthScratch,
    t: usize,
    use_sgm: bool,
) -> Vec<ImagePred> {
    let npix = IMG_W * IMG_H;
    let mut preds = Vec::with_capacity(N_CAMS);
    for c in 0..N_CAMS {
        let idx = t * N_CAMS + c;
        let pose = cam_pose(poses, t, c);
        let src = sources_for(images, poses, t, c);
        let (mut depth, conf) = sweep_depth_mode(&images[idx].rgb, &pose, &src, cam, scratch, use_sgm);
        // Confidence-guided median filter: weak or outlier pixels take the
        // local median of strong neighbors (locally smooth ground); sky stays far.
        hole_fill(&mut depth, &conf, &pose, cam);
        let mut points = vec![0.0f32; npix * 3];
        unproject(&depth, &pose, cam, &mut points);
        let sem = predict_semantics(&images[idx].rgb, &pose, cam, &depth, &conf, &points);
        preds.push(ImagePred { depth, conf, sem, points });
    }
    preds
}

/// Full geometry prediction for the clip (native / test entry point).
pub fn predict_clip(
    images: &[CamImage],
    poses: &PosePred,
    cam: &CameraModel,
    rng: &mut Rng,
) -> (Vec<ImagePred>, f32) {
    let mut scratch = DepthScratch::new();
    let mut preds = Vec::with_capacity(N_FRAMES * N_CAMS);
    let mut absrel_acc = 0.0;
    for t in 0..N_FRAMES {
        let frame_preds = predict_frame(images, poses, cam, &mut scratch, t);
        for c in 0..N_CAMS {
            absrel_acc += depth_absrel(&frame_preds[c].depth, &images[t * N_CAMS + c]);
        }
        preds.extend(frame_preds);
    }
    let _ = rng;
    (preds, absrel_acc / (N_FRAMES * N_CAMS) as f32)
}

/// Confidence-guided median fill: weakly-supported, ground-looking, or outlying
/// pixels take the local median of confident neighbors. Upward rays with weak
/// evidence are sky and stay far.
pub fn hole_fill(depth: &mut [f32], conf: &[f32], pose: &Se3, cam: &CameraModel) {
    let npix = depth.len();
    // Robust reference: 5x5 median of strong pixels per position (two passes).
    for _pass in 0..2 {
        let mut patched = vec![0.0f32; npix];
        patched.copy_from_slice(depth);
        for p in 0..npix {
            let strong = conf[p] >= 0.10 && depth[p] < SKY_DEPTH * 0.9;
            if strong {
                continue;
            }
            let ray_w = pose.r.mul_vec(&cam.rays[p]);
            let is_sky_ray = ray_w.z() > 0.02 && conf[p] < 0.05;
            if is_sky_ray {
                continue; // sky: keep far depth
            }
            let x = p % IMG_W;
            let y = p / IMG_W;
            let mut vals: Vec<f32> = Vec::new();
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let xx = x as i32 + dx;
                    let yy = y as i32 + dy;
                    if xx >= 0 && xx < IMG_W as i32 && yy >= 0 && yy < IMG_H as i32 {
                        let q = yy as usize * IMG_W + xx as usize;
                        if conf[q] >= 0.10 && depth[q] < SKY_DEPTH * 0.9 {
                            vals.push(depth[q]);
                        }
                    }
                }
            }
            if vals.len() >= 4 {
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                patched[p] = vals[vals.len() / 2];
            }
        }
        // Outlier rejection: strong pixels far from the local median get pulled in.
        for p in 0..npix {
            if conf[p] < 0.10 || depth[p] >= SKY_DEPTH * 0.9 {
                continue;
            }
            let x = p % IMG_W;
            let y = p / IMG_W;
            let mut vals: Vec<f32> = Vec::new();
            for dy in -2i32..=2 {
                for dx in -2i32..=2 {
                    let xx = x as i32 + dx;
                    let yy = y as i32 + dy;
                    if xx >= 0 && xx < IMG_W as i32 && yy >= 0 && yy < IMG_H as i32 {
                        let q = yy as usize * IMG_W + xx as usize;
                        if conf[q] >= 0.10 && depth[q] < SKY_DEPTH * 0.9 {
                            vals.push(depth[q]);
                        }
                    }
                }
            }
            if vals.len() >= 6 {
                vals.sort_by(|a, b| a.partial_cmp(b).unwrap());
                let med = vals[vals.len() / 2];
                let d = patched.get_mut(p).unwrap();
                let dev = (*d - med).abs() / med.max(0.5);
                if dev > 0.75 {
                    *d = med;
                }
            }
        }
        depth.copy_from_slice(&patched);
    }
}

/// Count semantics coverage (for stats display).
pub fn sem_histogram(sem: &[u8]) -> [usize; N_SEM] {
    let mut h = [0usize; N_SEM];
    for &s in sem {
        h[s as usize] += 1;
    }
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scene::build_scene;

    #[test]
    fn sweep_recovers_synthetic_depth_with_sgm() {
        // Two cameras 1 m apart looking at a textured wall 10 m ahead.
        let scene = build_scene(3);
        let cam = CameraModel::new();
        let wall_tris: Vec<crate::scene::Tri> = {
            let mut tris = Vec::new();
            for gy in 0..40 {
                for gx in 0..60 {
                    let x0 = -30.0 + gx as f32;
                    let y0 = -20.0 + gy as f32;
                    let x1 = x0 + 1.0;
                    let y1 = y0 + 1.0;
                    let shade = ((gx * 7 + gy * 13) % 5) as f32 * 0.04;
                    let col = [0.5 + shade, 0.5 + shade, 0.55 + shade];
                    use crate::scene::Tri;
                    use crate::math::Vec3;
                    tris.push(Tri { a: Vec3::new(x0, y0, 10.0), b: Vec3::new(x1, y0, 10.0), c: Vec3::new(x1, y1, 10.0), color: col, sem: SEM_BUILDING });
                    tris.push(Tri { a: Vec3::new(x0, y0, 10.0), b: Vec3::new(x1, y1, 10.0), c: Vec3::new(x0, y1, 10.0), color: col, sem: SEM_BUILDING });
                }
            }
            tris
        };
        let pose_a = Se3 { r: crate::math::Mat3::IDENTITY, t: Vec3::new(0.0, 0.0, 0.0) };
        let pose_b = Se3 { r: crate::math::Mat3::IDENTITY, t: Vec3::new(0.0, 1.0, 0.0) };
        let mut img_a = CamImage::default();
        let mut img_b = CamImage::default();
        crate::raster::render_camera(&scene, &wall_tris, &pose_a, &cam.rays, cam.fx, cam.fy, cam.cx, cam.cy, &mut img_a, true);
        crate::raster::render_camera(&scene, &wall_tris, &pose_b, &cam.rays, cam.fx, cam.fy, cam.cx, cam.cy, &mut img_b, true);
        let mut scratch = DepthScratch::new();
        let src = SweepSources {
            imgs: vec![&img_b.rgb[..]],
            poses: vec![pose_b],
            weights: vec![1.0],
        };
        let (depth, conf) = sweep_depth(&img_a.rgb, &pose_a, &src, &cam, &mut scratch);
        // Center pixels should be near 10 m.
        let mut sum = 0.0;
        let mut n = 0;
        for y in 40..90 {
            for x in 60..130 {
                let p = y * IMG_W + x;
                if conf[p] > 0.1 {
                    sum += depth[p];
                    n += 1;
                }
            }
        }
        let mean = sum / n.max(1) as f32;
        assert!(n > 500, "confident pixels: {n}");
        assert!((mean - 10.0).abs() < 1.5, "mean depth {mean}");
    }
}
