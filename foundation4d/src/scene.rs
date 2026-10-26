//! Procedural driving scene: ego rig with 6 surround cameras (60 deg apart, 80 deg FOV),
//! a curved road, textured buildings/trees, and moving actors (vehicles, bus,
//! pedestrians). Everything is deterministic given a seed.
//!
//! Frames follow the paper's setup: camera c=1 (index 0) defines reference timestamps,
//! keyframe rate 2.5 Hz (nuScenes-style).

use crate::math::{Mat3, Quat, Rng, Se3, Vec3};

pub const N_CAMS: usize = 6;
pub const IMG_W: usize = 192;
pub const IMG_H: usize = 144;
/// Longer clip: 32 keyframes at 2.5 Hz = 12.8 s of driving (~102 m at 8 m/s),
/// processed as 8-frame overlapping windows (paper Sec 5 video inference).
pub const N_FRAMES: usize = 32;
pub const DT: f32 = 0.4; // 2.5 Hz keyframes
pub const FOV_X: f32 = 80.0_f32.to_radians();

/// Semantic classes (subset of the paper's 21-class label space).
pub const SEM_UNKNOWN: u8 = 0;
pub const SEM_ROAD: u8 = 1;
pub const SEM_SIDEWALK: u8 = 2;
pub const SEM_BUILDING: u8 = 3;
pub const SEM_VEGETATION: u8 = 4;
pub const SEM_VEHICLE: u8 = 5;
pub const SEM_PERSON: u8 = 6;
pub const SEM_SKY: u8 = 7;
pub const N_SEM: usize = 8;

pub const CLASS_NAMES: [&str; N_SEM] = [
    "unknown", "road", "sidewalk", "building", "vegetation", "vehicle", "person", "sky",
];

/// RGB color prototypes (used by the semantic head stand-in for obstacle sub-classes).
pub const CLASS_PROTO: [[f32; 3]; N_SEM] = [
    [0.5, 0.5, 0.5],    // unknown
    [0.33, 0.34, 0.36], // road asphalt
    [0.55, 0.53, 0.50], // sidewalk
    [0.62, 0.58, 0.52], // building warm gray
    [0.25, 0.44, 0.20], // vegetation green
    [0.60, 0.35, 0.30], // vehicle (saturated family average)
    [0.22, 0.23, 0.27], // person dark clothing
    [0.55, 0.68, 0.85], // sky
];

/// A rendered camera image: RGB (u8), z-depth (f32, meters), semantics (u8).
#[derive(Clone)]
pub struct CamImage {
    pub rgb: Vec<u8>,
    pub depth: Vec<f32>,
    pub sem: Vec<u8>,
}

impl Default for CamImage {
    fn default() -> Self {
        CamImage {
            rgb: vec![0u8; IMG_W * IMG_H * 3],
            depth: vec![0.0f32; IMG_W * IMG_H],
            sem: vec![SEM_UNKNOWN; IMG_W * IMG_H],
        }
    }
}

/// A textured surface triangle in world space.
#[derive(Clone, Copy)]
pub struct Tri {
    pub a: Vec3,
    pub b: Vec3,
    pub c: Vec3,
    pub color: [f32; 3],
    pub sem: u8,
}

/// Moving actor with a cuboid body, moving ALONG the road (arc-length
/// parameterization keeps vehicles in their lanes as the road curves).
#[derive(Clone)]
pub struct Actor {
    pub id: usize,
    pub class: u8,        // SEM_VEHICLE or SEM_PERSON
    pub dims: Vec3,       // length (x), width (y), height (z) when yaw = 0
    pub s0: f32,          // road arc length at t = 0
    pub lat: f32,         // lateral offset from the centerline (+ = left)
    pub v: f32,           // longitudinal velocity along the road (m/s)
    pub color: [f32; 3],
    pub is_bus: bool,
}

impl Actor {
    /// World pose of the actor body frame at time t.
    pub fn pose(&self, t: f32) -> (Vec3, f32) {
        let s = self.s0 + self.v * t;
        let (c, yaw) = road_center(s);
        let n = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
        (c.add(&n.scale(self.lat)), yaw)
    }
}

pub struct Scene {
    pub tris: Vec<Tri>,
    pub actors: Vec<Actor>,
    /// Ground-truth rig poses (world-from-rig) at each keyframe.
    pub ego_gt: Vec<Se3>,
    /// Rig extrinsics: camera pose in rig frame (rig-from-camera is its inverse).
    pub cams_rig: [Se3; N_CAMS],
    /// Focal lengths / principal point (identical pinhole cameras).
    pub fx: f32,
    pub fy: f32,
    pub cx: f32,
    pub cy: f32,
    pub sun_dir: Vec3,
}

/// Road centerline: gentle S-curve. Returns (position, tangent yaw) at arc length s.
fn road_center(s: f32) -> (Vec3, f32) {
    // x runs forward, y lateral: y = A * sin(s / L)
    let (a, l) = (7.0, 55.0);
    let y = a * (s / l).sin();
    let dy = (a / l) * (s / l).cos();
    let yaw = dy.atan2(1.0);
    (Vec3::new(s, y, 0.0), yaw)
}

/// Add a grid of small triangles covering a quad on the ground (for stereo texture).
fn add_ground_patch(tris: &mut Vec<Tri>, x0: f32, x1: f32, y0: f32, y1: f32, color: [f32; 3], sem: u8, step: f32, z: f32) {
    let nx = ((x1 - x0) / step).ceil() as usize;
    let ny = ((y1 - y0) / step).ceil() as usize;
    for i in 0..nx {
        for j in 0..ny {
            let ax = x0 + i as f32 * step;
            let ay = y0 + j as f32 * step;
            let bx = (ax + step).min(x1);
            let by = (ay + step).min(y1);
            let slight = (i * 31 + j * 17) % 7;
            let c = [
                (color[0] + 0.012 * slight as f32).min(1.0),
                (color[1] + 0.011 * slight as f32).min(1.0),
                (color[2] + 0.013 * slight as f32).min(1.0),
            ];
            tris.push(Tri { a: Vec3::new(ax, ay, z), b: Vec3::new(bx, ay, z), c: Vec3::new(bx, by, z), color: c, sem });
            tris.push(Tri { a: Vec3::new(ax, ay, z), b: Vec3::new(bx, by, z), c: Vec3::new(ax, by, z), color: c, sem });
        }
    }
}

/// Add a box (axis-aligned or yawed) as textured triangles.
fn add_box(tris: &mut Vec<Tri>, center: &Vec3, dims: &Vec3, yaw: f32, color: [f32; 3], sem: u8, step: f32) {
    let r = Mat3::from_yaw(yaw);
    let hd = dims.scale(0.5);
    // 8 corners
    let mut pts = [Vec3::zero(); 8];
    let idx = [
        [-1.0, -1.0, -1.0], [1.0, -1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, 1.0, -1.0],
        [-1.0, -1.0, 1.0], [1.0, -1.0, 1.0], [1.0, 1.0, 1.0], [-1.0, 1.0, 1.0],
    ];
    for (k, s) in idx.iter().enumerate() {
        let local = Vec3::new(s[0] * hd.x(), s[1] * hd.y(), s[2] * hd.z());
        pts[k] = center.add(&r.mul_vec(&local));
    }
    // Faces: (a, b, c, d) quads with outward normals.
    let faces = [
        (4, 5, 6, 7, 0.0),  // top
        (0, 3, 2, 1, -0.14), // bottom (darker)
        (5, 1, 2, 6, 0.0),  // +y side
        (0, 4, 7, 3, 0.0),  // -y side
        (1, 0, 4, 5, -0.05), // -x face (front-ish darker)
        (3, 2, 6, 7, 0.05), // +x face
    ];
    let (dx, dy, dz) = (hd.x() * 2.0, hd.y() * 2.0, hd.z() * 2.0);
    for (f, (ia, ib, ic, id, shade)) in faces.into_iter().enumerate() {
        let base = [0.0, dy, dy, dz][(f.min(1) + if f == 0 { 1 } else { 0 }) as usize].max(0.35);
        let dim_l = [dx, dy, dy, dz][if f >= 4 { 3 } else if f >= 2 { (f - 2) as usize % 2 + 1 } else { 0 }];
        let dim_w = if f == 0 || f == 1 { dx.min(dy) } else if f >= 4 { dy } else { dx };
        let _ = (base, dim_l, dim_w);
        let face_color = [
            (color[0] + shade).clamp(0.0, 1.0),
            (color[1] + shade).clamp(0.0, 1.0),
            (color[2] + shade).clamp(0.0, 1.0),
        ];
        // Subdivide the quad for stereo-matchable texture variation.
        let n1 = 2usize.max(1);
        let n2 = 2usize.max(1);
        let corner = |u: f32, v: f32| -> Vec3 {
            let p01 = pts[ia].add(&pts[ib].sub(&pts[ia]).scale(u));
            let p03 = pts[ia].add(&pts[id].sub(&pts[ia]).scale(v));
            p01.add(&p03.sub(&pts[ia]))
        };
        for i in 0..n1 {
            for j in 0..n2 {
                let u0 = i as f32 / n1 as f32;
                let u1 = (i + 1) as f32 / n1 as f32;
                let v0 = j as f32 / n2 as f32;
                let v1 = (j + 1) as f32 / n2 as f32;
                let jitter = ((i + j + f) % 3) as f32 * 0.015;
                let c = [
                    (face_color[0] + jitter).min(1.0),
                    (face_color[1] + jitter).min(1.0),
                    (face_color[2] + jitter).min(1.0),
                ];
                let _ = step;
                tris.push(Tri { a: corner(u0, v0), b: corner(u1, v0), c: corner(u1, v1), color: c, sem });
                tris.push(Tri { a: corner(u0, v0), b: corner(u1, v1), c: corner(u0, v1), color: c, sem });
            }
        }
    }
}

pub fn build_scene(seed: u64) -> Scene {
    let mut rng = Rng::new(seed ^ 0xF4D0);

    // ---- Ego trajectory: drives along the road centerline at ~8 m/s. ----
    let speed = 8.0;
    let s0 = -6.0;
    let mut ego_gt = Vec::with_capacity(N_FRAMES);
    for t in 0..N_FRAMES {
        let s = s0 + speed * (t as f32 * DT);
        let (p, yaw) = road_center(s);
        // Rig frame: x forward along tangent, y left, z up.
        let r = Mat3::from_yaw(yaw);
        ego_gt.push(Se3 { r, t: Vec3::new(p.x(), p.y(), 0.0) });
    }

    // ---- Camera rig: 6 cameras at 60 deg yaw increments, 80 deg FOV. ----
    let mut cams_rig = [Se3::IDENTITY; N_CAMS];
    let pitch_down = -7.0_f32.to_radians();
    for (c, cam) in cams_rig.iter_mut().enumerate() {
        let yaw = c as f32 * (core::f32::consts::PI / 3.0);
        let r = cam_basis(yaw, pitch_down);
        let pos = Vec3::new(yaw.cos() * 1.1, yaw.sin() * 1.1, 1.45);
        *cam = Se3 { r, t: pos };
    }

    // ---- Static geometry ----
    let mut tris: Vec<Tri> = Vec::with_capacity(90_000);

    // Road strip along the centerline (width 8 m), subdivided along s.
    // NOTE: lane markings are deliberately NOT drawn as thin geometry — sub-pixel
    // wide stripes alias inconsistently across views and destroy multi-view matching.
    // The road's photometric signal comes from the smooth world-space value noise
    // plus a gentle 2 m longitudinal shading.
    let road_half = 4.0;
    let mut s = -35.0f32;
    let ds = 1.0;
    while s < 140.0 {
        let (p0, _) = road_center(s);
        let (p1, _) = road_center(s + ds);
        // Left/right extents (normal direction).
        let tangent = p1.sub(&p0).normalized();
        let normal = Vec3::new(-tangent.y(), tangent.x(), 0.0);
        // Road surface (asphalt) with gentle longitudinal shading.
        let shade = ((s / (2.0 * ds)) as i64 % 2) as f32 * 0.008;
        let col = [0.32 + shade, 0.33 + shade, 0.35 + shade];
        let a = p0.add(&normal.scale(-road_half));
        let b = p0.add(&normal.scale(road_half));
        let c2 = p1.add(&normal.scale(road_half));
        let d2 = p1.add(&normal.scale(-road_half));
        tris.push(Tri { a, b, c: c2, color: col, sem: SEM_ROAD });
        tris.push(Tri { a, c: c2, b: d2, color: col, sem: SEM_ROAD });

        // Sidewalks (raised 0.12 m) and grass beyond.
        for side in [-1.0f32, 1.0] {
            let sw_in = road_half + 0.05;
            let sw_out = road_half + 2.4;
            let g_out = road_half + 26.0;
            let sa0 = p0.add(&normal.scale(side * sw_in));
            let sb0 = p0.add(&normal.scale(side * sw_out));
            let sa1 = p1.add(&normal.scale(side * sw_in));
            let sb1 = p1.add(&normal.scale(side * sw_out));
            let zsw = 0.12;
            let colsw = [0.55, 0.53, 0.50];
            tris.push(Tri { a: sa0.add(&Vec3::new(0.0, 0.0, zsw)), b: sb0.add(&Vec3::new(0.0, 0.0, zsw)), c: sb1.add(&Vec3::new(0.0, 0.0, zsw)), color: colsw, sem: SEM_SIDEWALK });
            tris.push(Tri { a: sa0.add(&Vec3::new(0.0, 0.0, zsw)), c: sb1.add(&Vec3::new(0.0, 0.0, zsw)), b: sa1.add(&Vec3::new(0.0, 0.0, zsw)), color: colsw, sem: SEM_SIDEWALK });
            // Kerb (vertical face).
            let colk = [0.48, 0.46, 0.44];
            tris.push(Tri { a: sa0, b: sa0.add(&Vec3::new(0.0, 0.0, zsw)), c: sa1.add(&Vec3::new(0.0, 0.0, zsw)), color: colk, sem: SEM_SIDEWALK });
            tris.push(Tri { a: sa0, c: sa1.add(&Vec3::new(0.0, 0.0, zsw)), b: sa1, color: colk, sem: SEM_SIDEWALK });
            // Grass.
            let colg = [0.24, 0.40, 0.19];
            let ga0 = p0.add(&normal.scale(side * sw_out));
            let gb0 = p0.add(&normal.scale(side * g_out));
            let ga1 = p1.add(&normal.scale(side * sw_out));
            let gb1 = p1.add(&normal.scale(side * g_out));
            let shade_g = ((s * 3.7).sin() * 0.5 + 0.5) * 0.03;
            let colg2 = [colg[0] + shade_g, colg[1] + shade_g, colg[2] + shade_g * 0.5];
            tris.push(Tri { a: ga0, b: gb0, c: gb1, color: colg2, sem: SEM_VEGETATION });
            tris.push(Tri { a: ga0, c: gb1, b: ga1, color: colg2, sem: SEM_VEGETATION });
        }
        s += ds;
    }

    // Buildings: boxes along both sides at 11-16 m from centerline.
    let mut bs = -28.0f32;
    let mut bid = 0;
    while bs < 138.0 {
        let (p, yaw) = road_center(bs);
        let tangent_yaw = yaw;
        let n = Vec3::new(-tangent_yaw.sin(), tangent_yaw.cos(), 0.0);
        for side in [-1.0f32, 1.0] {
            if rng.f32() < 0.18 { continue; } // occasional gap
            let dist = 11.5 + rng.f32() * 4.5;
            let w = 4.5 + rng.f32() * 6.0;
            let hgt = 4.0 + rng.f32() * 9.0;
            let depth = 5.0 + rng.f32() * 4.0;
            let c = p.add(&n.scale(side * (dist + depth * 0.5))).add(&Vec3::new(0.0, 0.0, hgt * 0.5));
            // Warm gray / tan palette (desaturated family for prototype separability).
            let base = 0.5 + rng.f32() * 0.18;
            let color = [base + 0.06, base, base - 0.05];
            add_box(&mut tris, &c, &Vec3::new(depth, w, hgt), tangent_yaw, color, SEM_BUILDING, 1.0);
            bid += 1;
        }
        bs += 9.0 + rng.f32() * 6.0;
    }
    let _ = bid;

    // Trees on the sidewalk edge every ~9 m, alternating sides.
    let mut ts = -20.0f32;
    while ts < 132.0 {
        let (p, yaw) = road_center(ts);
        let n = Vec3::new(-yaw.sin(), yaw.cos(), 0.0);
        let side = if ((ts / 9.0) as i64) % 2 == 0 { 1.0 } else { -1.0 };
        let base = p.add(&n.scale(side * 6.6));
        // Trunk.
        add_box(&mut tris, &base.add(&Vec3::new(0.0, 0.0, 1.1)), &Vec3::new(0.35, 0.35, 2.2), 0.0, [0.35, 0.27, 0.18], SEM_VEGETATION, 0.3);
        // Foliage.
        let fg = 2.0 + rng.f32() * 0.8;
        let gc = [0.20 + rng.f32() * 0.1, 0.38 + rng.f32() * 0.12, 0.16];
        add_box(&mut tris, &base.add(&Vec3::new(0.0, 0.0, 2.2 + fg * 0.35)), &Vec3::new(fg, fg, fg * 1.1), 0.0, gc, SEM_VEGETATION, 0.6);
        ts += 9.0;
    }

    // ---- Actors ----
    // Traffic designed for close-range multi-view observation: the ego catches up
    // to slower lead vehicles, oncoming traffic passes alongside, and pedestrians
    // walk the near sidewalks — exercising detection, tracking and actor-aware
    // Gaussian reconstruction throughout the clip.
    let mut actors: Vec<Actor> = Vec::new();
    let vehicle_colors = [
        [0.72, 0.22, 0.19], [0.20, 0.35, 0.62], [0.85, 0.66, 0.16], [0.88, 0.88, 0.90],
        [0.25, 0.55, 0.35], [0.55, 0.25, 0.55], [0.30, 0.30, 0.33], [0.90, 0.45, 0.15],
    ];
    // Same-direction vehicles in the +y lanes (lead traffic at varying gaps).
    let same_dir = [
        (10.0, -2.8, 5.5),  // slow lead the ego overtakes mid-clip
        (24.0, 1.9, 7.6),   // steady mid-range lead
        (42.0, 1.9, 8.6),   // pulls slowly away
        (-10.0, -1.9, 9.2), // behind, slowly caught by the ego
        (58.0, 1.85, 7.8),
    ];
    for (i, (ahead, lat, v)) in same_dir.iter().enumerate() {
        actors.push(Actor {
            id: i,
            class: SEM_VEHICLE,
            dims: Vec3::new(4.4, 1.85, 1.5),
            s0: 8.0 + ahead,
            lat: *lat,
            v: *v,
            color: vehicle_colors[i % vehicle_colors.len()],
            is_bus: false,
        });
    }
    // Oncoming traffic in the -y lanes (passes alongside the ego mid-clip).
    let oncoming = [(35.0, -2.1, -7.0), (60.0, -5.4, -6.5), (15.0, -2.1, -8.0)];
    for (i, (ahead, lat, v)) in oncoming.iter().enumerate() {
        actors.push(Actor {
            id: 10 + i,
            class: SEM_VEHICLE,
            dims: Vec3::new(4.6, 1.9, 1.6),
            s0: 8.0 + ahead,
            lat: *lat,
            v: *v,
            color: vehicle_colors[(i + 4) % vehicle_colors.len()],
            is_bus: false,
        });
    }
    // A bus ahead that the ego gradually catches.
    actors.push(Actor {
        id: 20,
        class: SEM_VEHICLE,
        dims: Vec3::new(9.5, 2.45, 3.0),
        s0: 8.0 + 16.0,
        lat: 1.9,
        v: 6.2,
        color: [0.85, 0.55, 0.12],
        is_bus: true,
    });
    // Pedestrians on the near sidewalks.
    let peds = [(4.0, 1, 1.2), (10.0, -1, -1.1), (18.0, 1, 1.3), (28.0, -1, 1.0)];
    for (i, (ahead, side, v)) in peds.iter().enumerate() {
        actors.push(Actor {
            id: 30 + i,
            class: SEM_PERSON,
            dims: Vec3::new(0.45, 0.42, 1.75),
            s0: 8.0 + ahead,
            lat: *side as f32 * 6.2,
            v: *v * *side as f32,
            color: [0.18 + 0.08 * (i as f32), 0.20, 0.26],
            is_bus: false,
        });
    }

    // Base ground plane far under everything (slightly below the road surface to
    // avoid z-fighting with the road strip; only visible beyond the grass verge).
    add_ground_patch(&mut tris, -60.0, 170.0, -70.0, 70.0, [0.24, 0.40, 0.19], SEM_VEGETATION, 4.0, -0.02);

    let sun_dir = Vec3::new(0.45, 0.35, 0.82).normalized();

    Scene { tris, actors, ego_gt, cams_rig, fx: 0.0, fy: 0.0, cx: 0.0, cy: 0.0, sun_dir }
}

/// Camera basis in rig frame: +z forward (yaw), +x right, +y down, tilted by pitch.
fn cam_basis(yaw: f32, pitch: f32) -> Mat3 {
    // Columns are the camera axes expressed in the rig frame.
    let x_cam = Vec3::new(yaw.sin(), -yaw.cos(), 0.0);
    let y_cam = Vec3::new(0.0, 0.0, -1.0);
    let z_cam = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
    let rx = Mat3([x_cam.x(), y_cam.x(), z_cam.x(),
                   x_cam.y(), y_cam.y(), z_cam.y(),
                   x_cam.z(), y_cam.z(), z_cam.z()]);
    // Pitch (tilt down): rotate about the camera x-axis.
    rx.mul_mat(&Mat3::from_pitch(pitch))
}

/// World pose of camera (t, c) given the rig trajectory.
pub fn cam_pose_world(scene: &Scene, t: usize, c: usize) -> Se3 {
    scene.ego_gt[t].compose(&scene.cams_rig[c])
}

/// Actor triangles at time t (for rasterization).
pub fn actor_tris(actor: &Actor, t: f32, out: &mut Vec<Tri>) {
    let (pos, yaw) = actor.pose(t);
    let center = Vec3::new(pos.x(), pos.y(), actor.dims.z() * 0.5);
    let mut tris = Vec::new();
    add_box(&mut tris, &center, &actor.dims, yaw, actor.color, actor.class, 0.3);
    if actor.is_bus {
        // Windows band on the bus sides for texture variety.
        let r = Mat3::from_yaw(yaw);
        for side in [-1.0f32, 1.0] {
            let local = Vec3::new(0.0, side * (actor.dims.y() * 0.5 + 0.01), 1.9);
            let wc = center.add(&r.mul_vec(&local));
            add_box(&mut tris, &wc, &Vec3::new(actor.dims.x() * 0.8, 0.02, 0.7), yaw, [0.55, 0.62, 0.68], SEM_VEHICLE, 0.3);
        }
    }
    out.extend(tris);
}
