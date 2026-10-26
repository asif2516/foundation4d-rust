//! Differentiable-style Gaussian splatting rasterizer (paper Sec 3.7 scene rendering):
//! tile-based (16x16) EWA projection (covariance Sigma' = J W Sigma W^T J^T), global
//! depth ordering via a counting sort, front-to-back alpha compositing with
//! transmittance, sky cubemap fill for the residual transmittance (Eq. 10 with
//! identity per-camera affine color), and depth-tested 3D overlay drawing
//! (actor cuboids, ego trajectory, rig marker).
//!
//! All buffers are preallocated and reused across frames — render calls do not
//! allocate (except growing the entry buffer, amortized).

use crate::gauss::{Gaussians, SkyCube, track_state_at};
use crate::math::{Mat3, Se3, Vec3};
use crate::scene::{N_CAMS, DT};
use crate::tracking::Track;

pub const TILE: usize = 16;

pub struct Renderer {
    pub w: usize,
    pub h: usize,
    pub rgba: Vec<u8>,     // w*h*4 output
    depth: Vec<f32>,       // first-hit depth per pixel
    cbuf: Vec<f32>,        // w*h*3 color accumulator
    tbuf: Vec<f32>,        // transmittance
    rays: Vec<f32>,        // w*h*3 world rays (per camera)
    // Cached view basis for overlay drawing.
    eye_cache: Vec3,
    fwd_cache: Vec3,
    right_cache: Vec3,
    down_cache: Vec3,
    fov_cache: f32,
    // Per-Gaussian scratch (sized to capacity, reused).
    gz: Vec<f32>,
    gu: Vec<f32>,
    gv: Vec<f32>,
    gia: Vec<f32>, // inverse covariance entries
    gib: Vec<f32>,
    gic: Vec<f32>,
    gru: Vec<f32>, // 3-sigma extents
    grv: Vec<f32>,
    gcolor: Vec<f32>, // per gaussian 3
    galpha: Vec<f32>,
    // Tile entries: (key, tile, gidx) packed into u64.
    entries: Vec<u64>,
    counts: Vec<u32>,
    offsets: Vec<u32>,
    sorted: Vec<u64>,
    pub drawn: usize,
    pub blends: u64,
    pub last_ms: f32,
    #[cfg(feature = "splat_dbg")]
    pub dbg_wsum: Vec<f32>,
    #[cfg(feature = "splat_dbg")]
    pub dbg_proj: usize,
    #[cfg(feature = "splat_dbg")]
    pub dbg_med_ru: f32,
    #[cfg(feature = "splat_dbg")]
    pub dbg_near: usize,
}

impl Renderer {
    pub fn new(w: usize, h: usize) -> Renderer {
        let mut r = Renderer {
            w: 1,
            h: 1,
            rgba: vec![0; 4],
            depth: vec![0.0; 1],
            cbuf: vec![0.0; 3],
            tbuf: vec![0.0; 1],
            rays: vec![0.0; 3],
            eye_cache: Vec3::zero(),
            fwd_cache: Vec3::new(0.0, 0.0, 1.0),
            right_cache: Vec3::new(1.0, 0.0, 0.0),
            down_cache: Vec3::new(0.0, -1.0, 0.0),
            fov_cache: 1.0,
            gz: Vec::new(),
            gu: Vec::new(),
            gv: Vec::new(),
            gia: Vec::new(),
            gib: Vec::new(),
            gic: Vec::new(),
            gru: Vec::new(),
            grv: Vec::new(),
            gcolor: Vec::new(),
            galpha: Vec::new(),
            entries: Vec::new(),
            counts: vec![0; 65536],
            offsets: vec![0; 65536],
            sorted: Vec::new(),
            drawn: 0,
            blends: 0,
            last_ms: 0.0,
            #[cfg(feature = "splat_dbg")]
            dbg_wsum: vec![0.0; w.max(16) * h.max(16)],
            #[cfg(feature = "splat_dbg")]
            dbg_proj: 0,
            #[cfg(feature = "splat_dbg")]
            dbg_med_ru: 0.0,
            #[cfg(feature = "splat_dbg")]
            dbg_near: 0,
        };
        r.set_size(w, h);
        r
    }

    pub fn set_size(&mut self, w: usize, h: usize) {
        self.w = w.max(16);
        self.h = h.max(16);
        #[cfg(feature = "splat_dbg")]
        {
            self.dbg_wsum = vec![0.0; self.w * self.h];
        }
        let n = self.w * self.h;
        self.rgba = vec![255; n * 4];
        self.depth = vec![200.0; n];
        self.cbuf = vec![0.0; n * 3];
        self.tbuf = vec![1.0; n];
        self.rays = vec![0.0; n * 3];
    }

    /// Render the scene at time `tau` from a camera at `eye` looking along `fwd`
    /// (world frame, up = +Z), vertical FOV `fov_y` radians.
    #[allow(clippy::too_many_arguments)]
    pub fn render(
        &mut self,
        g: &Gaussians,
        sky: &SkyCube,
        tracks: &[Track],
        assembled: &[Se3],
        rig: &[Se3],
        eye: [f32; 3],
        fwd: [f32; 3],
        fov_y: f32,
        tau: f32,
        draw_overlays: u32,
    ) {
        let n = self.w * self.h;
        // Reset buffers.
        for v in self.tbuf.iter_mut() { *v = 1.0; }
        for v in self.cbuf.iter_mut() { *v = 0.0; }
        for v in self.depth.iter_mut() { *v = 200.0; }

        let eye = Vec3::new(eye[0], eye[1], eye[2]);
        let fwd = Vec3::new(fwd[0], fwd[1], fwd[2]).normalized();
        let up_w = Vec3::new(0.0, 0.0, 1.0);
        let right = fwd.cross(&up_w).normalized();
        let down = right.cross(&fwd).normalized();
        self.eye_cache = eye;
        self.fwd_cache = fwd;
        self.right_cache = right;
        self.down_cache = down;
        self.fov_cache = fov_y;
        let fy = (self.h as f32 * 0.5) / (fov_y * 0.5).tan();
        let fx = fy; // square pixels
        let cx = self.w as f32 * 0.5;
        let cy = self.h as f32 * 0.5;

        // Per-pixel world rays (for the sky fill).
        for y in 0..self.h {
            for x in 0..self.w {
                let sx = (x as f32 + 0.5 - cx) / fx;
                let sy = (y as f32 + 0.5 - cy) / fy;
                let d = fwd.add(&right.scale(sx)).add(&down.scale(sy)).normalized();
                let i = (y * self.w + x) * 3;
                self.rays[i] = d.x();
                self.rays[i + 1] = d.y();
                self.rays[i + 2] = d.z();
            }
        }

        // Actor transforms at tau.
        let mut actor_r: Vec<Mat3> = Vec::with_capacity(tracks.len());
        let mut actor_t: Vec<Vec3> = Vec::with_capacity(tracks.len());
        for tr in tracks {
            let (c, yaw) = track_state_at(tr, tau);
            actor_r.push(Mat3::from_yaw(yaw));
            actor_t.push(Vec3::new(c[0], c[1], c[2] + tr.dims[2] * 0.5));
        }

        // ---- Project all Gaussians, emit tile entries. ----
        let ntiles_x = (self.w + TILE - 1) / TILE;
        let ntiles_y = (self.h + TILE - 1) / TILE;
        let ntiles = ntiles_x * ntiles_y;
        let _ = ntiles;
        self.entries.clear();
        // Grow scratch if needed.
        let gn = g.n;
        if self.gz.len() < gn {
            self.gz.resize(gn, 0.0);
            self.gu.resize(gn, 0.0);
            self.gv.resize(gn, 0.0);
            self.gia.resize(gn, 0.0);
            self.gib.resize(gn, 0.0);
            self.gic.resize(gn, 0.0);
            self.gru.resize(gn, 0.0);
            self.grv.resize(gn, 0.0);
            self.gcolor.resize(gn * 3, 0.0);
            self.galpha.resize(gn, 0.0);
        }
        let far = 95.0f32;
        let margin = 32.0f32;
        for i in 0..gn {
            let a = g.actor[i];
            let (wx, wy, wz) = if a >= 0 {
                let ai = a as usize;
                let l = Vec3::new(g.pos[i * 3], g.pos[i * 3 + 1], g.pos[i * 3 + 2]);
                let p = actor_r[ai].mul_vec(&l).add(&actor_t[ai]);
                (p.x(), p.y(), p.z())
            } else {
                (g.pos[i * 3], g.pos[i * 3 + 1], g.pos[i * 3 + 2])
            };
            let d = Vec3::new(wx - eye.x(), wy - eye.y(), wz - eye.z());
            let z = d.dot(&fwd);
            if z < 1.2 || z > far {
                continue;
            }
            let x = d.dot(&right);
            let y = d.dot(&down);
            let u = cx + fx * x / z;
            let v = cy + fy * y / z;
            // EWA covariance: Sigma' = J W R S R^T W^T J^T.
            // M = W R (rows: right, down, fwd).
            let q = &g.quat[i * 4..i * 4 + 4];
            let (qw, qx, qy, qz) = (q[0], q[1], q[2], q[3]);
            // R columns: apply quaternion rotation to basis via matrix directly.
            let r00 = 1.0 - 2.0 * (qy * qy + qz * qz);
            let r01 = 2.0 * (qx * qy - qw * qz);
            let r02 = 2.0 * (qx * qz + qw * qy);
            let r10 = 2.0 * (qx * qy + qw * qz);
            let r11 = 1.0 - 2.0 * (qx * qx + qz * qz);
            let r12 = 2.0 * (qy * qz - qw * qx);
            let r20 = 2.0 * (qx * qz - qw * qy);
            let r21 = 2.0 * (qy * qz + qw * qx);
            let r22 = 1.0 - 2.0 * (qx * qx + qy * qy);
            // M = W (3x3) * R: W rows are right/down/fwd.
            let (w0x, w0y, w0z) = (right.x(), right.y(), right.z());
            let (w1x, w1y, w1z) = (down.x(), down.y(), down.z());
            let (w2x, w2y, w2z) = (fwd.x(), fwd.y(), fwd.z());
            let m00 = w0x * r00 + w0y * r10 + w0z * r20;
            let m01 = w0x * r01 + w0y * r11 + w0z * r21;
            let m02 = w0x * r02 + w0y * r12 + w0z * r22;
            let m10 = w1x * r00 + w1y * r10 + w1z * r20;
            let m11 = w1x * r01 + w1y * r11 + w1z * r21;
            let m12 = w1x * r02 + w1y * r12 + w1z * r22;
            let m20 = w2x * r00 + w2y * r10 + w2z * r20;
            let m21 = w2x * r01 + w2y * r11 + w2z * r21;
            let m22 = w2x * r02 + w2y * r12 + w2z * r22;
            // Sigma_cam = M diag(s^2) M^T (symmetric).
            let (s0, s1, s2) = (g.scale[i * 3], g.scale[i * 3 + 1], g.scale[i * 3 + 2]);
            let (a0, a1, a2) = (s0 * s0, s1 * s1, s2 * s2);
            let sc00 = m00 * m00 * a0 + m01 * m01 * a1 + m02 * m02 * a2;
            let sc11 = m10 * m10 * a0 + m11 * m11 * a1 + m12 * m12 * a2;
            let sc22 = m20 * m20 * a0 + m21 * m21 * a1 + m22 * m22 * a2;
            let sc01 = m00 * m10 * a0 + m01 * m11 * a1 + m02 * m12 * a2;
            let sc02 = m00 * m20 * a0 + m01 * m21 * a1 + m02 * m22 * a2;
            let sc12 = m10 * m20 * a0 + m11 * m21 * a1 + m12 * m22 * a2;
            // J = [[fx/z, 0, -fx x / z^2], [0, fy/z, -fy y / z^2]].
            let invz = 1.0 / z;
            let j00 = fx * invz;
            let j02 = -fx * x * invz * invz;
            let j11 = fy * invz;
            let j12 = -fy * y * invz * invz;
            let sa = j00 * j00 * sc00 + j02 * j02 * sc22 + 2.0 * j00 * j02 * sc02;
            let sb = j00 * j11 * sc01 + j00 * j12 * sc02 + j02 * j11 * sc12 + j02 * j12 * sc22;
            let sc = j11 * j11 * sc11 + j12 * j12 * sc22 + 2.0 * j11 * j12 * sc12;
            // Clamp the projected Gaussian size: surfels sourced from far-away
            // cameras carry large footprints and would render as giant blobs when
            // viewed up close (no LOD in this demo) — cap sigma at ~2.5 px.
            let smax = sa.max(sc);
            const MAX_SIGMA2: f32 = 2.5 * 2.5;
            let (sa, sb, sc) = if smax > MAX_SIGMA2 {
                let f = MAX_SIGMA2 / smax;
                (sa * f, sb * f, sc * f)
            } else {
                (sa, sb, sc)
            };
            let det = sa * sc - sb * sb;
            if det <= 1e-12 || !det.is_finite() {
                continue;
            }
            let inv_det = 1.0 / det;
            self.gz[i] = z;
            self.gu[i] = u;
            self.gv[i] = v;
            self.gia[i] = sc * inv_det;
            self.gib[i] = -sb * inv_det;
            self.gic[i] = sa * inv_det;
            // 3-sigma extents from the covariance diagonal.
            let ru = 3.0 * sa.sqrt();
            let rv = 3.0 * sc.sqrt();
            self.gru[i] = ru;
            self.grv[i] = rv;
            self.gcolor[i * 3] = g.color[i * 3] as f32;
            self.gcolor[i * 3 + 1] = g.color[i * 3 + 1] as f32;
            self.gcolor[i * 3 + 2] = g.color[i * 3 + 2] as f32;
            self.galpha[i] = g.opacity[i];
            if u + ru < -margin || u - ru > self.w as f32 + margin
                || v + rv < -margin || v - rv > self.h as f32 + margin {
                continue;
            }
            // Emit tile entries.
            let tx0 = (((u - ru).max(0.0) / TILE as f32).floor() as i32).max(0);
            let tx1 = (((u + ru) / TILE as f32).floor() as i32).min(ntiles_x as i32 - 1);
            let ty0 = (((v - rv).max(0.0) / TILE as f32).floor() as i32).max(0);
            let ty1 = (((v + rv) / TILE as f32).floor() as i32).min(ntiles_y as i32 - 1);
            if tx1 < tx0 || ty1 < ty0 {
                continue;
            }
            let key = ((z / far * 65535.0).clamp(0.0, 65535.0) as u64) as u16;
            for ty in ty0..=ty1 {
                for tx in tx0..=tx1 {
                    let tile = (ty * ntiles_x as i32 + tx) as u64;
                    self.entries.push((key as u64) << 48 | tile << 32 | i as u64);
                }
            }
        }
        self.drawn = self.entries.len();
        #[cfg(feature = "splat_dbg")]
        {
            let mut n_proj = 0usize;
            let mut sum_ru = 0.0f32;
            let mut n_near = 0usize;
            for i in 0..gn {
                if self.gz[i] == 0.0 && self.gu[i] == 0.0 { continue; }
                n_proj += 1;
                sum_ru += self.gru[i];
                if self.gz[i] < 10.0 { n_near += 1; }
            }
            self.dbg_proj = n_proj;
            self.dbg_med_ru = if n_proj > 0 { sum_ru / n_proj as f32 } else { 0.0 };
            self.dbg_near = n_near;
        }

        // ---- Counting sort by depth key (16-bit), stable in emission order. ----
        for c in self.counts.iter_mut() { *c = 0; }
        for &e in &self.entries {
            self.counts[(e >> 48) as usize] += 1;
        }
        let mut acc = 0u32;
        for k in 0..65536 {
            self.offsets[k] = acc;
            acc += self.counts[k];
        }
        if self.sorted.len() < self.entries.len() {
            self.sorted.resize(self.entries.len(), 0);
        }
        let mut cursor = self.offsets.clone();
        for &e in &self.entries {
            let k = (e >> 48) as usize;
            self.sorted[cursor[k] as usize] = e;
            cursor[k] += 1;
        }

        // ---- Front-to-back per-tile blending. ----
        self.blends = 0;
        let ne = self.entries.len();
        for ei in 0..ne {
            let e = self.sorted[ei];
            let i = (e & 0xFFFFFFFF) as usize;
            let tile = ((e >> 32) & 0xFFFF) as usize;
            let tx = tile % ntiles_x;
            let ty = tile / ntiles_x;
            let x0 = (tx * TILE).max(0) as i32;
            let y0 = (ty * TILE) as i32;
            let x1 = (x0 + TILE as i32).min(self.w as i32);
            let y1 = (y0 + TILE as i32).min(self.h as i32);
            let u = self.gu[i];
            let v = self.gv[i];
            let ia = self.gia[i];
            let ib = self.gib[i];
            let ic = self.gic[i];
            let alpha0 = self.galpha[i];
            let cr = self.gcolor[i * 3];
            let cg = self.gcolor[i * 3 + 1];
            let cb = self.gcolor[i * 3 + 2];
            let z = self.gz[i];
            // Pixel bounds: tile ∩ 3-sigma AABB.
            let px0 = (u - self.gru[i]).ceil().max(x0 as f32) as i32;
            let px1 = (u + self.gru[i]).floor().min(x1 as f32 - 1.0) as i32;
            let py0 = (v - self.grv[i]).ceil().max(y0 as f32) as i32;
            let py1 = (v + self.grv[i]).floor().min(y1 as f32 - 1.0) as i32;
            if px1 < px0 || py1 < py0 {
                continue;
            }
            for yy in py0..=py1 {
                let dv = yy as f32 + 0.5 - v;
                let row = yy as usize * self.w;
                for xx in px0..=px1 {
                    let du = xx as f32 + 0.5 - u;
                    let qv = ia * du * du + 2.0 * ib * du * dv + ic * dv * dv;
                    if qv > 9.0 {
                        continue;
                    }
                    let p = row + xx as usize;
                    let tr = self.tbuf[p];
                    if tr < 0.02 {
                        continue;
                    }
                    let alpha = alpha0 * (-0.5 * qv).exp();
                    if alpha < 0.01 {
                        continue;
                    }
                    let wgt = tr * alpha;
                    let c3 = &mut self.cbuf[p * 3..p * 3 + 3];
                    c3[0] += wgt * cr;
                    c3[1] += wgt * cg;
                    c3[2] += wgt * cb;
                    self.tbuf[p] = tr * (1.0 - alpha);
                    #[cfg(feature = "splat_dbg")]
                    {
                        self.dbg_wsum[p] += wgt;
                    }
                    if alpha > 0.25 && self.depth[p] > z {
                        if self.depth[p] >= 199.0 {
                            self.depth[p] = z;
                        }
                    }
                    self.blends += 1;
                }
            }
        }

        // ---- Sky fill + tone-map to RGBA. ----
        for p in 0..n {
            let tr = self.tbuf[p];
            let (mut r, mut g2, mut b) = (self.cbuf[p * 3], self.cbuf[p * 3 + 1], self.cbuf[p * 3 + 2]);
            if tr > 0.02 {
                let ray = Vec3::new(self.rays[p * 3], self.rays[p * 3 + 1], self.rays[p * 3 + 2]);
                let sc = sky.sample(&ray);
                r += tr * sc[0] as f32;
                g2 += tr * sc[1] as f32;
                b += tr * sc[2] as f32;
            }
            self.rgba[p * 4] = r.clamp(0.0, 255.0) as u8;
            self.rgba[p * 4 + 1] = g2.clamp(0.0, 255.0) as u8;
            self.rgba[p * 4 + 2] = b.clamp(0.0, 255.0) as u8;
            self.rgba[p * 4 + 3] = 255;
        }

        // ---- Overlays ----
        if draw_overlays & 1 != 0 {
            // Actor cuboids at tau.
            for tr in tracks {
                let (c, yaw) = track_state_at(tr, tau);
                let col: [u8; 3] = if tr.class == crate::scene::SEM_PERSON {
                    [255, 120, 60]
                } else {
                    [80, 220, 120]
                };
                draw_cuboid(self, &Vec3::new(c[0], c[1], c[2]), &tr.dims, yaw, col);
            }
        }
        if draw_overlays & 2 != 0 {
            // Ego trajectory + rig marker at the nearest frame.
            let col = [255, 210, 60];
            for k in 1..assembled.len() {
                draw_line(self, &assembled[k - 1].t, &assembled[k].t, col);
            }
            let fi = (tau / DT).round().clamp(0.0, (assembled.len() - 1) as f32) as usize;
            let pose = assembled[fi];
            draw_rig(self, &pose, rig);
        }
        if draw_overlays & 4 != 0 {
            // Current-frame camera frustums.
            let fi = (tau / DT).round().clamp(0.0, (assembled.len() - 1) as f32) as usize;
            let pose = assembled[fi];
            for c in 0..N_CAMS {
                let cp = pose.compose(&rig[c]);
                draw_frustum(self, &cp);
            }
        }
    }
}

/// Project a world point to screen using the renderer's cached view basis.
/// Returns (u, v, z) or None if behind the camera.
#[inline]
fn project_cached(r: &Renderer, p: &Vec3) -> Option<(f32, f32, f32)> {
    let d = p.sub(&r.eye_cache);
    let z = d.dot(&r.fwd_cache);
    if z < 0.2 {
        return None;
    }
    let x = d.dot(&r.right_cache);
    let y = d.dot(&r.down_cache);
    let fy = (r.h as f32 * 0.5) / (r.fov_cache * 0.5).tan();
    let u = r.w as f32 * 0.5 + fy * x / z;
    let v = r.h as f32 * 0.5 + fy * y / z;
    Some((u, v, z))
}

fn draw_line(r: &mut Renderer, a: &Vec3, b: &Vec3, col: [u8; 3]) {
    let (ua, va, za) = match project_cached(r, a) {
        Some(v) => v,
        None => return,
    };
    let (ub, vb, zb) = match project_cached(r, b) {
        Some(v) => v,
        None => return,
    };
    // Bresenham with depth test.
    let (x0, y0) = (ua.round() as i32, va.round() as i32);
    let (x1, y1) = (ub.round() as i32, vb.round() as i32);
    let dx = (x1 - x0).abs();
    let dy = -(y1 - y0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;
    let mut x = x0;
    let mut y = y0;
    let steps = (dx.abs().max(dy.abs()).max(1)) as usize;
    for k in 0..=steps {
        let a_t = k as f32 / steps as f32;
        let z = za + (zb - za) * a_t;
        if x >= 0 && x < r.w as i32 && y >= 0 && y < r.h as i32 {
            let p = y as usize * r.w + x as usize;
            if z <= r.depth[p] + 0.5 {
                r.rgba[p * 4] = col[0];
                r.rgba[p * 4 + 1] = col[1];
                r.rgba[p * 4 + 2] = col[2];
            }
        }
        if x == x1 && y == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x += sx;
        }
        if e2 <= dx {
            err += dx;
            y += sy;
        }
    }
}

fn draw_cuboid(r: &mut Renderer, center: &Vec3, dims: &[f32; 3], yaw: f32, col: [u8; 3]) {
    let rr = Mat3::from_yaw(yaw);
    let hd = Vec3::new(dims[0] * 0.5, dims[1] * 0.5, dims[2] * 0.5);
    let c = Vec3::new(center.x(), center.y(), center.z() + dims[2] * 0.5);
    let mut pts = [Vec3::zero(); 8];
    for k in 0..8 {
        let sx = if k & 1 != 0 { 1.0 } else { -1.0 };
        let sy = if k & 2 != 0 { 1.0 } else { -1.0 };
        let sz = if k & 4 != 0 { 1.0 } else { -1.0 };
        let local = Vec3::new(sx * hd.x(), sy * hd.y(), sz * hd.z());
        pts[k] = c.add(&rr.mul_vec(&local));
    }
    let edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ];
    for (a, b) in edges {
        draw_line(r, &pts[a], &pts[b], col);
    }
}

fn draw_rig(r: &mut Renderer, pose: &Se3, rig: &[Se3]) {
    // Small axis marker at the rig origin.
    let o = pose.t;
    let ax_x = pose.r.mul_vec(&Vec3::new(1.2, 0.0, 0.0)).add(&o);
    let ax_y = pose.r.mul_vec(&Vec3::new(0.0, 1.2, 0.0)).add(&o);
    let ax_z = pose.r.mul_vec(&Vec3::new(0.0, 0.0, 1.2)).add(&o);
    draw_line(r, &o, &ax_x, [255, 80, 80]);
    draw_line(r, &o, &ax_y, [80, 255, 80]);
    draw_line(r, &o, &ax_z, [80, 80, 255]);
    // Camera positions.
    for c in rig {
        let cp = pose.compose(c);
        draw_line(r, &cp.t, &cp.t.add(&Vec3::new(0.0, 0.0, 0.35)), [200, 200, 220]);
    }
}

fn draw_frustum(r: &mut Renderer, cam: &Se3) {
    // Small pyramid showing the camera's view direction.
    let o = cam.t;
    let tip = cam.r.mul_vec(&Vec3::new(0.0, 0.0, 0.9)).add(&o);
    let fw = cam.r.mul_vec(&Vec3::new(0.0, 0.0, 0.9));
    let _ = fw;
    for (dx, dy) in [(-0.3, -0.22), (0.3, -0.22), (0.3, 0.22), (-0.3, 0.22)] {
        let corner = cam.r.mul_vec(&Vec3::new(dx, dy, 0.7)).add(&o);
        draw_line(r, &tip, &corner, [120, 140, 255]);
    }
    draw_line(r, &o, &tip, [120, 140, 255]);
}

