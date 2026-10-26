//! Gaussian reconstruction (paper Sec 3.7): per-source-pixel 3D Gaussians decoded
//! from depth + assembled camera poses (Eq. 9: mu = R (d / r_z) r + t), with
//! color taken from the resampled source RGB, opacity gated by the semantic mask
//! (sky / unknown suppressed, as in the paper), covariance Sigma = R diag(s^2) R^T
//! from surface-aligned scales, and actor association through the tracked cuboids.
//! Static Gaussians stay fixed; actor Gaussians are stored in actor-local frames and
//! rigidly transformed to render target timestamps (paper's scene rendering).
//! Sky pixels fill a 6-face cubemap that supplies the residual transmittance color
//! (Eq. 10 without the per-camera affine transform).

use crate::geometry::{CameraModel, ImagePred};
use crate::math::{Mat3, Quat, Se3, Vec3};
use crate::raster::SKY_DEPTH;
use crate::scene::{
    CamImage, CLASS_NAMES, N_CAMS, N_FRAMES, SEM_SKY, SEM_UNKNOWN, SEM_VEGETATION,
    SEM_PERSON, SEM_VEHICLE, DT, IMG_H, IMG_W,
};
use crate::tracking::Track;

pub const SKY_FACE: usize = 16;

/// SoA Gaussian cloud.
pub struct Gaussians {
    pub n: usize,
    /// For static Gaussians: world position. For actor Gaussians: actor-local
    /// position at the source capture time.
    pub pos: Vec<f32>,        // 3n
    pub quat: Vec<f32>,       // 4n (w, x, y, z)
    pub scale: Vec<f32>,      // 3n
    pub color: Vec<u8>,       // 3n
    pub opacity: Vec<f32>,    // n
    pub actor: Vec<i32>,      // n: track index or -1
    pub src_time: Vec<f32>,   // n: capture time (seconds)
}

/// 6-face sky cubemap (RGB u8), +X, -X, +Y, -Y, +Z, -Z faces.
pub struct SkyCube {
    pub faces: Vec<u8>, // 6 * FACE^2 * 3
}

impl SkyCube {
    pub fn new() -> SkyCube {
        // Default: vertical gradient per face.
        let mut faces = vec![128u8; 6 * SKY_FACE * SKY_FACE * 3];
        for f in 0..6 {
            for v in 0..SKY_FACE {
                let t = v as f32 / SKY_FACE as f32;
                let (r, g, b) = (
                    (200.0 - 60.0 * t) as u8,
                    (215.0 - 60.0 * t) as u8,
                    (235.0 - 60.0 * t) as u8,
                );
                for u in 0..SKY_FACE {
                    let idx = (f * SKY_FACE * SKY_FACE + v * SKY_FACE + u) * 3;
                    faces[idx] = r;
                    faces[idx + 1] = g;
                    faces[idx + 2] = b;
                }
            }
        }
        SkyCube { faces }
    }

    /// Accumulate an observed sky color along a world ray.
    pub fn add(&mut self, ray: &Vec3, rgb: [u8; 3]) {
        let (ax, ay, az) = (ray.x().abs(), ray.y().abs(), ray.z().abs());
        let (face, u, v): (usize, f32, f32) = if ax >= ay && ax >= az {
            if ray.x() > 0.0 {
                (0, ray.y() / ax, ray.z() / ax)
            } else {
                (1, -ray.y() / ax, ray.z() / ax)
            }
        } else if ay >= ax && ay >= az {
            if ray.y() > 0.0 {
                (2, -ray.x() / ay, ray.z() / ay)
            } else {
                (3, ray.x() / ay, ray.z() / ay)
            }
        } else if ray.z() > 0.0 {
            (4, ray.x() / az, ray.y() / az)
        } else {
            (5, ray.x() / az, -ray.y() / az)
        };
        let uu = ((u * 0.5 + 0.5) * (SKY_FACE - 1) as f32).round() as usize;
        let vv = ((v * 0.5 + 0.5) * (SKY_FACE - 1) as f32).round() as usize;
        let idx = (face * SKY_FACE * SKY_FACE + vv * SKY_FACE + uu) * 3;
        // Running average.
        for c in 0..3 {
            let cur = self.faces[idx + c] as u16;
            let add = rgb[c] as u16;
            self.faces[idx + c] = ((cur + add) / 2) as u8;
        }
    }

    /// Sample the cubemap along a world ray (nearest face texel).
    pub fn sample(&self, ray: &Vec3) -> [u8; 3] {
        let (ax, ay, az) = (ray.x().abs(), ray.y().abs(), ray.z().abs());
        let (face, u, v): (usize, f32, f32) = if ax >= ay && ax >= az {
            if ray.x() > 0.0 { (0, ray.y() / ax, ray.z() / ax) } else { (1, -ray.y() / ax, ray.z() / ax) }
        } else if ay >= ax && ay >= az {
            if ray.y() > 0.0 { (2, -ray.x() / ay, ray.z() / ay) } else { (3, ray.x() / ay, ray.z() / ay) }
        } else if ray.z() > 0.0 {
            (4, ray.x() / az, ray.y() / az)
        } else {
            (5, ray.x() / az, -ray.y() / az)
        };
        let uu = ((u * 0.5 + 0.5) * (SKY_FACE - 1) as f32).round() as usize;
        let vv = ((v * 0.5 + 0.5) * (SKY_FACE - 1) as f32).round() as usize;
        let idx = (face * SKY_FACE * SKY_FACE + vv * SKY_FACE + uu) * 3;
        [self.faces[idx], self.faces[idx + 1], self.faces[idx + 2]]
    }
}

/// Is a point inside an actor's cuboid at time t?
fn inside_cuboid(p: &Vec3, center: &[f32; 3], dims: &[f32; 3], yaw: f32, margin: f32) -> bool {
    let dx = p.x() - center[0];
    let dy = p.y() - center[1];
    let dz = p.z() - center[2];
    let (c, s) = (yaw.cos(), yaw.sin());
    // Rotate into the cuboid frame (inverse yaw).
    let lx = c * dx + s * dy;
    let ly = -s * dx + c * dy;
    lx.abs() <= dims[0] * 0.5 + margin
        && ly.abs() <= dims[1] * 0.5 + margin
        && dz.abs() <= dims[2] * 0.5 + margin + 0.15
}

/// Track state (position + yaw) at a continuous time, with constant-velocity
/// extrapolation outside the observed interval.
pub fn track_state_at(track: &Track, tau: f32) -> ([f32; 3], f32) {
    let s = &track.states;
    if s.is_empty() {
        return ([0.0; 3], 0.0);
    }
    if tau <= s[0][0] * DT {
        let st = s[0];
        let dt = tau - st[0] * DT;
        return ([st[1] + st[3] * dt, st[2] + st[4] * dt, 0.75], st[5]);
    }
    if tau >= s[s.len() - 1][0] * DT {
        let st = s[s.len() - 1];
        let dt = tau - st[0] * DT;
        return ([st[1] + st[3] * dt, st[2] + st[4] * dt, 0.75], st[5]);
    }
    for i in 1..s.len() {
        let t1 = s[i][0] * DT;
        if tau <= t1 {
            let t0 = s[i - 1][0] * DT;
            let a = (tau - t0) / (t1 - t0).max(1e-6);
            let x = s[i - 1][1] + (s[i][1] - s[i - 1][1]) * a;
            let y = s[i - 1][2] + (s[i][2] - s[i - 1][2]) * a;
            let yaw = s[i - 1][5] + (s[i][5] - s[i - 1][5]) * a;
            return ([x, y, 0.75], yaw);
        }
    }
    ([s[0][1], s[0][2], 0.75], s[0][5])
}

/// Decode the full Gaussian cloud (paper Eq. 9 + Sec 3.7).
#[allow(clippy::too_many_arguments)]
pub fn decode_gaussians(
    preds: &[ImagePred],
    images: &[CamImage],
    assembled: &[Se3],   // optimized reference-camera trajectory per frame (clip frame)
    rig: &[Se3],         // rig transforms (camera c in rig frame)
    tracks: &[Track],
    cam: &CameraModel,
    stride: usize,
) -> (Gaussians, SkyCube) {
    let cam_fx = (IMG_W as f32 * 0.5) / (crate::scene::FOV_X * 0.5).tan();
    let mut sky = SkyCube::new();
    let mut pos: Vec<f32> = Vec::with_capacity(200_000 * 3);
    let mut quat: Vec<f32> = Vec::with_capacity(200_000 * 4);
    let mut scale: Vec<f32> = Vec::with_capacity(200_000 * 3);
    let mut color: Vec<u8> = Vec::with_capacity(200_000 * 3);
    let mut opacity: Vec<f32> = Vec::with_capacity(200_000);
    let mut actor: Vec<i32> = Vec::with_capacity(200_000);
    let mut src_time: Vec<f32> = Vec::with_capacity(200_000);

    for t in 0..N_FRAMES {
        let tau = t as f32 * DT;
        // Per-frame actor poses (for local-frame encoding).
        let actor_poses: Vec<([f32; 3], f32)> = tracks.iter().map(|tr| track_state_at(tr, tau)).collect();
        for c in 0..N_CAMS {
            let idx = t * N_CAMS + c;
            let img = &images[idx];
            let pred = &preds[idx];
            // Assembled camera pose: T*_t composed with the rig transform (Eq. 16,
            // synchronized case).
            let pose = assembled[t].compose(&rig[c]);
            let r = &pose.r;
            let ro = &pose.t;
            for py in (0..IMG_H - 1).step_by(stride) {
                for px in (0..IMG_W - 1).step_by(stride) {
                    let p = py * IMG_W + px;
                    let sem = pred.sem[p];
                    let ray = &cam.rays[p];
                    // Semantic opacity mask (paper: multiply opacities by the mask
                    // that suppresses sky / ego / unknown regions).
                    if sem == SEM_SKY {
                        // Contribute to the sky cubemap instead.
                        let ray_w = pose.r.mul_vec(ray);
                        sky.add(&ray_w, [img.rgb[p * 3], img.rgb[p * 3 + 1], img.rgb[p * 3 + 2]]);
                        continue;
                    }
                    if sem == SEM_UNKNOWN || pred.conf[p] < 0.03 || pred.depth[p] >= SKY_DEPTH * 0.9 {
                        continue;
                    }
                    let d = pred.depth[p];
                    // Eq. 9: mu = R (d / r_z) r + t.
                    let s = d / ray.z();
                    let wx = r.0[0] * ray.x() * s + r.0[1] * ray.y() * s + r.0[2] * ray.z() * s + ro.x();
                    let wy = r.0[3] * ray.x() * s + r.0[4] * ray.y() * s + r.0[5] * ray.z() * s + ro.y();
                    let wz = r.0[6] * ray.x() * s + r.0[7] * ray.y() * s + r.0[8] * ray.z() * s + ro.z();
                    // Surface-aligned anisotropic scales from depth gradients.
                    let d_r = pred.depth[p + 1];
                    let d_d = pred.depth[p + IMG_W];
                    let ray_r = &cam.rays[p + 1];
                    let ray_d = &cam.rays[p + IMG_W];
                    let un = |rr: &Vec3, dd: f32| -> Vec3 {
                        let ss = dd / rr.z();
                        Vec3::new(
                            r.0[0] * rr.x() * ss + r.0[1] * rr.y() * ss + r.0[2] * rr.z() * ss + ro.x(),
                            r.0[3] * rr.x() * ss + r.0[4] * rr.y() * ss + r.0[5] * rr.z() * ss + ro.y(),
                            r.0[6] * rr.x() * ss + r.0[7] * rr.y() * ss + r.0[8] * rr.z() * ss + ro.z(),
                        )
                    };
                    let p0 = Vec3::new(wx, wy, wz);
                    let pr = un(ray_r, d_r);
                    let pd = un(ray_d, d_d);
                    let t1 = pr.sub(&p0);
                    let t2 = pd.sub(&p0);
                    let nrm = {
                        let mut nn = t2.cross(&t1);
                        if nn.norm() < 1e-9 {
                            nn = Vec3::new(0.0, 0.0, 1.0);
                        }
                        let mut nn = nn.normalized();
                        // Orient toward the camera for a front-facing surfel.
                        let view = ro.sub(&p0);
                        if nn.dot(&view) < 0.0 {
                            nn = nn.scale(-1.0);
                        }
                        nn
                    };
                    let mut u1 = t1;
                    if u1.norm() < 1e-9 {
                        u1 = Vec3::new(1.0, 0.0, 0.0);
                    }
                    let u1 = u1.normalized();
                    let u2 = nrm.cross(&u1).normalized();
                    // R columns = [u1, u2, n].
                    let rot = Mat3([
                        u1.x(), u2.x(), nrm.x(),
                        u1.y(), u2.y(), nrm.y(),
                        u1.z(), u2.z(), nrm.z(),
                    ]);
                    let q = Quat::from_mat(&rot);
                    // Surfel in-plane scale from the SOURCE pixel footprint: adjacent
                    // decoded Gaussians are `stride` source pixels apart, so a disk of
                    // ~0.75x that spacing (3-sigma overlap) guarantees hole-free
                    // coverage from any view — raw neighbor-tangent lengths explode
                    // on grazing surfaces (walls, oblique road) and turn the surfels
                    // into giant blobs.
                    let footprint = stride as f32 * d / cam_fx;
                    let s_tan = (footprint * 0.75).clamp(0.05, 1.2);
                    let s1 = s_tan;
                    let s2 = s_tan;
                    let s3 = 0.055;
                    // Actor association: which tracked cuboid contains this point?
                    let world = Vec3::new(wx, wy, wz);
                    let mut assigned: i32 = -1;
                    for (ti, tr) in tracks.iter().enumerate() {
                        let (ac, ayaw) = actor_poses[ti];
                        if inside_cuboid(&world, &ac, &tr.dims, ayaw, 0.1) {
                            assigned = ti as i32;
                            break;
                        }
                    }
                    // Store in the actor-local frame when associated.
                    let (sx, sy, sz) = if assigned >= 0 {
                        let (ac, ayaw) = actor_poses[assigned as usize];
                        let rr = Mat3::from_yaw(ayaw);
                        let local = rr.transpose().mul_vec(&world.sub(&Vec3::new(ac[0], ac[1], ac[2])));
                        (local.x(), local.y(), local.z())
                    } else {
                        (wx, wy, wz)
                    };
                    pos.extend_from_slice(&[sx, sy, sz]);
                    quat.extend_from_slice(&q.0);
                    scale.extend_from_slice(&[s1, s2, s3]);
                    color.extend_from_slice(&[
                        img.rgb[p * 3],
                        img.rgb[p * 3 + 1],
                        img.rgb[p * 3 + 2],
                    ]);
                    // Surface surfels are effectively opaque; confidence modulates
                    // transparency only mildly (single-layer coverage means low
                    // opacity would let the sky bleed through the whole scene).
                    let base = if sem == SEM_VEGETATION { 0.88 } else { 0.97 };
                    let alpha = (base * (0.7 + 0.3 * pred.conf[p])).clamp(0.45, 0.98);
                    opacity.push(alpha);
                    actor.push(assigned);
                    src_time.push(tau);
                }
            }
        }
    }
    let n = opacity.len();
    (
        Gaussians { n, pos, quat, scale, color, opacity, actor, src_time },
        sky,
    )
}

/// Static-gaussian semantic check helper used by stats.
pub fn count_semantics(g: &Gaussians) -> usize {
    g.n
}

#[allow(unused)]
fn _unused(a: &[&str]) -> usize {
    CLASS_NAMES.len() + SEM_PERSON as usize + SEM_VEHICLE as usize
}
