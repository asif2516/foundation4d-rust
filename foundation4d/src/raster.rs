//! Tiny software rasterizer: renders the procedural scene into each surround camera
//! (RGB + z-depth + per-pixel semantics). Triangles are near-plane clipped, projected
//! with a pinhole model, and depth-tested. Uncovered pixels receive the procedural sky.
//! A view-independent hash-noise pass adds high-frequency surface texture so multi-view
//! stereo has a photometric signal to match.

use crate::math::{hash3, Se3, Vec3};
use crate::scene::{CamImage, Scene, Tri, IMG_H, IMG_W, N_FRAMES, SEM_SKY};

pub const ZNEAR: f32 = 0.45;
pub const SKY_DEPTH: f32 = 400.0;

/// Smooth 3D value noise on a cubic lattice (trilinearly interpolated hash).
/// View-independent and continuous: the same surface point yields the same value in
/// every view, and the field is smooth at pixel scale so bilinear image sampling
/// stays accurate — giving plane-sweep stereo a clean, discriminative cost minimum.
#[inline]
fn value_noise3(x: f32, y: f32, z: f32, lattice: f32) -> f32 {
    let inv = 1.0 / lattice;
    let fx = x * inv;
    let fy = y * inv;
    let fz = z * inv;
    let xi = fx.floor();
    let yi = fy.floor();
    let zi = fz.floor();
    let tx = fx - xi;
    let ty = fy - yi;
    let tz = fz - zi;
    // Smoothstep interpolation weights.
    let sx = tx * tx * (3.0 - 2.0 * tx);
    let sy = ty * ty * (3.0 - 2.0 * ty);
    let sz = tz * tz * (3.0 - 2.0 * tz);
    let c00 = hash3(xi, yi, zi) * (1.0 - sx) + hash3(xi + 1.0, yi, zi) * sx;
    let c10 = hash3(xi, yi + 1.0, zi) * (1.0 - sx) + hash3(xi + 1.0, yi + 1.0, zi) * sx;
    let c01 = hash3(xi, yi, zi + 1.0) * (1.0 - sx) + hash3(xi + 1.0, yi, zi + 1.0) * sx;
    let c11 = hash3(xi, yi + 1.0, zi + 1.0) * (1.0 - sx) + hash3(xi + 1.0, yi + 1.0, zi + 1.0) * sx;
    let c0 = c00 * (1.0 - sy) + c10 * sy;
    let c1 = c01 * (1.0 - sy) + c11 * sy;
    c0 * (1.0 - sz) + c1 * sz
}

/// Surface texture: two octaves of value noise (coarse structure + fine detail)
/// with per-channel gains, added to the flat base color.
#[inline]
fn surface_texture(x: f32, y: f32, z: f32) -> [f32; 3] {
    let n1 = value_noise3(x, y, z, 0.6) - 0.5;
    let n2 = value_noise3(x + 31.7, y - 17.3, z + 11.9, 0.16) - 0.5;
    let n = n1 * 0.78 + n2 * 0.22;
    [n * 52.0, n * 44.0, n * 48.0]
}

/// Precompute per-pixel camera-frame rays (shared intrinsics across the rig).
pub fn precompute_rays(fx: f32, fy: f32, cx: f32, cy: f32) -> Vec<Vec3> {
    let mut rays = Vec::with_capacity(IMG_W * IMG_H);
    for v in 0..IMG_H {
        for u in 0..IMG_W {
            let x = (u as f32 + 0.5 - cx) / fx;
            let y = (v as f32 + 0.5 - cy) / fy;
            rays.push(Vec3::new(x, y, 1.0).normalized());
        }
    }
    rays
}

/// Procedural sky color for a world-frame ray direction.
pub fn sky_color(ray: &Vec3, sun_dir: &Vec3) -> [f32; 3] {
    let elev = ray.z().clamp(-0.1, 1.0);
    let zen = [0.30, 0.45, 0.78];
    let hor = [0.74, 0.80, 0.88];
    let t = elev.clamp(0.0, 1.0).powf(0.55);
    let mut c = [
        hor[0] + t * (zen[0] - hor[0]),
        hor[1] + t * (zen[1] - hor[1]),
        hor[2] + t * (zen[2] - hor[2]),
    ];
    // Sun glow.
    let d = ray.dot(sun_dir).max(0.0);
    let glow = d.powf(180.0) * 0.9 + d.powf(8.0) * 0.18;
    c[0] += glow * 1.0;
    c[1] += glow * 0.85;
    c[2] += glow * 0.6;
    c
}

/// Rasterize all triangles visible from `cam_from_world` into `img`.
pub fn render_camera(
    scene: &Scene,
    tris: &[Tri],
    cam_world: &Se3,
    rays: &[Vec3],
    fx: f32,
    fy: f32,
    cx: f32,
    cy: f32,
    img: &mut CamImage,
    apply_noise: bool,
) {
    let inv = cam_world.inverse();
    let r = &inv.r;
    let t = &inv.t;
    let n = IMG_W * IMG_H;
    let mut zbuf = vec![f32::INFINITY; n];

    for tri in tris.iter() {
        // Transform to camera space (apply the inverse pose: R^T p + t_inv).
        let mut vc = [Vec3::zero(); 3];
        let mut zc = [0.0f32; 3];
        for (k, p) in [&tri.a, &tri.b, &tri.c].into_iter().enumerate() {
            let pc = r.mul_vec(p).add(t);
            zc[k] = pc.z();
            vc[k] = pc;
        }
        // Cheap reject: fully beyond 90 m or fully behind near plane.
        if zc[0] > 90.0 && zc[1] > 90.0 && zc[2] > 90.0 { continue; }
        if zc[0] < ZNEAR && zc[1] < ZNEAR && zc[2] < ZNEAR { continue; }

        // Near-plane clip (Sutherland-Hodgman on z >= ZNEAR).
        let mut poly: Vec<Vec3> = Vec::with_capacity(4);
        for k in 0..3 {
            let a = vc[k];
            let b = vc[(k + 1) % 3];
            let za = zc[k] >= ZNEAR;
            let zb = zc[(k + 1) % 3] >= ZNEAR;
            if za { poly.push(a); }
            if za != zb {
                let s = (ZNEAR - a.z()) / (b.z() - a.z());
                poly.push(a.add(&b.sub(&a).scale(s)));
            }
        }
        if poly.len() < 3 { continue; }

        // Project to screen.
        let mut px = [0.0f32; 4];
        let mut py = [0.0f32; 4];
        let mut iz = [0.0f32; 4];
        for (k, p) in poly.iter().enumerate() {
            if p.z() <= 0.0 { poly.clear(); break; }
            let invz = 1.0 / p.z();
            px[k] = fx * p.x() * invz + cx - 0.5;
            py[k] = fy * p.y() * invz + cy - 0.5;
            iz[k] = invz;
        }
        if poly.len() < 3 { continue; }

        // Fan triangulation of the clipped polygon.
        for f in 0..(poly.len() - 2) {
            let (x0, y0, z0i) = (px[0], py[0], iz[0]);
            let (x1, y1, z1i) = (px[f + 1], py[f + 1], iz[f + 1]);
            let (x2, y2, z2i) = (px[f + 2], py[f + 2], iz[f + 2]);
            let area = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0);
            if area.abs() < 1e-9 { continue; }
            // Ensure consistent orientation (no backface cull; depth test handles it).
            let (x2s, y2s, z2is) = if area < 0.0 { (x1, y1, z1i) } else { (x2, y2, z2i) };
            let (x1s, y1s, z1is) = if area < 0.0 { (x2, y2, z2i) } else { (x1, y1, z1i) };
            let a2 = (x1s - x0) * (y2s - y0) - (x2s - x0) * (y1s - y0);
            let inv_a2 = 1.0 / a2;

            let minx = x0.min(x1s).min(x2s).floor().max(0.0) as i32;
            let maxx = (x0.max(x1s).max(x2s).ceil() as i32).min(IMG_W as i32 - 1);
            let miny = y0.min(y1s).min(y2s).floor().max(0.0) as i32;
            let maxy = (y0.max(y1s).max(y2s).ceil() as i32).min(IMG_H as i32 - 1);
            if minx > maxx || miny > maxy { continue; }

            let e01 = (x1s - x0) * (y2s - y0) - (x2s - x0) * (y1s - y0);
            if e01 <= 0.0 { continue; }

            let col = tri.color;
            let sem = tri.sem;
            for yy in miny..=maxy {
                let sy = yy as f32 + 0.5;
                let row_off = (yy as usize) * IMG_W;
                for xx in minx..=maxx {
                    let sx = xx as f32 + 0.5;
                    // Barycentric weights via edge functions: each weight belongs to
                    // the vertex OPPOSITE its edge (e_a <-> vertex A, etc.).
                    // Perspective-correct depth interpolates 1/z affinely.
                    let e_a = ((x2s - x1s) * (sy - y1s) - (sx - x1s) * (y2s - y1s)) * inv_a2;
                    let e_b = ((sx - x0) * (y2s - y0) - (x2s - x0) * (sy - y0)) * inv_a2;
                    let e_c = ((x1s - x0) * (sy - y0) - (sx - x0) * (y1s - y0)) * inv_a2;
                    if e_a < 0.0 || e_b < 0.0 || e_c < 0.0 { continue; }
                    let invz = e_a * z0i + e_b * z1is + e_c * z2is;
                    let z = 1.0 / invz;
                    let idx = row_off + xx as usize;
                    if z < zbuf[idx] {
                        zbuf[idx] = z;
                        let p3 = &mut img.rgb[idx * 3..idx * 3 + 3];
                        p3[0] = (col[0] * 255.0).clamp(0.0, 255.0) as u8;
                        p3[1] = (col[1] * 255.0).clamp(0.0, 255.0) as u8;
                        p3[2] = (col[2] * 255.0).clamp(0.0, 255.0) as u8;
                        img.sem[idx] = sem;
                        img.depth[idx] = z;
                    }
                }
            }
        }
    }

    // Sky + noise pass.
    let cam_pos = cam_world.t;
    for idx in 0..n {
        let ray_cam = &rays[idx];
        let ray_world = cam_world.r.mul_vec(ray_cam);
        if zbuf[idx].is_infinite() {
            let c = sky_color(&ray_world, &scene.sun_dir);
            img.rgb[idx * 3] = (c[0] * 255.0) as u8;
            img.rgb[idx * 3 + 1] = (c[1] * 255.0) as u8;
            img.rgb[idx * 3 + 2] = (c[2] * 255.0) as u8;
            img.sem[idx] = SEM_SKY;
            img.depth[idx] = SKY_DEPTH;
        } else if apply_noise {
            // World-space deterministic noise (same surface point -> same value in every view).
            let z = img.depth[idx];
            let dist = z / ray_cam.z();
            let p = cam_pos.add(&ray_world.scale(dist));
            let e = surface_texture(p.x(), p.y(), p.z());
            img.rgb[idx * 3] = ((img.rgb[idx * 3] as f32) + e[0]).clamp(0.0, 255.0) as u8;
            img.rgb[idx * 3 + 1] = ((img.rgb[idx * 3 + 1] as f32) + e[1]).clamp(0.0, 255.0) as u8;
            img.rgb[idx * 3 + 2] = ((img.rgb[idx * 3 + 2] as f32) + e[2]).clamp(0.0, 255.0) as u8;
        }
    }
}

/// Assemble the full triangle buffer (static scene + actors at frame t).
pub fn frame_tris(scene: &Scene, t: usize) -> Vec<Tri> {
    let mut tris = scene.tris.clone();
    for a in &scene.actors {
        crate::scene::actor_tris(a, t as f32 * crate::scene::DT, &mut tris);
    }
    tris
}

/// Render the entire clip (all frames, all cameras).
pub fn render_clip(scene: &Scene, out: &mut Vec<CamImage>) {
    let fx = (IMG_W as f32 * 0.5) / (crate::scene::FOV_X * 0.5).tan();
    let fy = fx;
    let cx = IMG_W as f32 * 0.5;
    let cy = IMG_H as f32 * 0.5;
    let rays = precompute_rays(fx, fy, cx, cy);
    out.clear();
    out.reserve(crate::scene::N_CAMS * N_FRAMES);
    for t in 0..N_FRAMES {
        let tris = frame_tris(scene, t);
        for c in 0..crate::scene::N_CAMS {
            let pose = crate::scene::cam_pose_world(scene, t, c);
            let mut img = CamImage::default();
            render_camera(scene, &tris, &pose, &rays, fx, fy, cx, cy, &mut img, true);
            out.push(img);
        }
    }
}
