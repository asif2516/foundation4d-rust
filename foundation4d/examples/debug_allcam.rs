use foundation4d::geometry::{self, CameraModel, PosePred};
use foundation4d::math::{Mat3, Rng, Se3, Vec3};
use foundation4d::pgo;
use foundation4d::raster;
use foundation4d::scene::{build_scene, CLASS_NAMES, N_CAMS};

fn main() {
    let scene = build_scene(42);
    let cam = CameraModel::new();
    let mut images = Vec::new();
    raster::render_clip(&scene, &mut images);
    let gt_ref: Vec<Se3> = scene.ego_gt.iter().map(|p| Se3::between(&scene.ego_gt[0], p)).collect();
    let mut rng = Rng::new(42u64 ^ 0xABCD);
    let windows = pgo::simulate_window_predictions(&gt_ref, 4, 2, &mut rng);
    let mut retained = vec![Se3::IDENTITY; 16];
    for i in 0..16 {
        let mut best_w = 0; let mut best_d = f32::INFINITY;
        for (k, w) in windows.iter().enumerate() {
            if let Some(pos) = w.frames.iter().position(|&f| f == i) {
                let mid = (w.frames.len() - 1) as f32 * 0.5;
                let d = (pos as f32 - mid).abs();
                if d < best_d { best_d = d; best_w = k; }
            }
        }
        if let Some(pos) = windows[best_w].frames.iter().position(|&f| f == i) {
            retained[i] = windows[best_w].poses[pos];
        }
    }
    // Noisy rig like the pipeline.
    let mut rng2 = Rng::new(42u64 ^ 0xABCD);
    let _ = &mut rng2;
    let rig_noisy: Vec<Se3> = scene.cams_rig.iter().map(|c| {
        let axis = Vec3::new(rng2.normal_clamped(1.0), rng2.normal_clamped(1.0), rng2.normal_clamped(1.0)).normalized();
        let rr = Mat3::from_axis_angle(&axis, rng2.normal_clamped(0.002));
        let tt = Vec3::new(rng2.normal_clamped(0.01), rng2.normal_clamped(0.01), rng2.normal_clamped(0.005));
        Se3 { r: c.r.mul_mat(&rr), t: c.t.add(&c.r.mul_vec(&tt)) }
    }).collect();
    let mode = std::env::args().nth(1).unwrap_or_default();
    let use_gt_rig = mode == "gtrig";
    let use_gt_ego = mode == "gtall";
    let pp = PosePred {
        ego: if use_gt_ego { gt_ref.clone() } else { retained },
        rig: if use_gt_rig || use_gt_ego { scene.cams_rig.to_vec() } else { rig_noisy },
    };
    let (preds, absrel) = geometry::predict_clip(&images, &pp, &cam, &mut Rng::new(1));
    println!("overall AbsRel = {:.3}", absrel);
    for c in 0..N_CAMS {
        let mut acc = 0.0;
        for t in 0..16 {
            acc += geometry::depth_absrel(&preds[t * N_CAMS + c].depth, &images[t * N_CAMS + c]);
        }
        println!("cam {} ({}): mean AbsRel = {:.3}", c, cam_name(c), acc / 16.0);
    }
    // Also per-frame for the worst camera.
    let mut worst_c = 0; let mut worst = 0.0;
    for c in 0..N_CAMS {
        let mut acc = 0.0;
        for t in 0..16 { acc += geometry::depth_absrel(&preds[t * N_CAMS + c].depth, &images[t * N_CAMS + c]); }
        if acc / 16.0 > worst { worst = acc / 16.0; worst_c = c; }
    }
    println!("worst cam = {} ({})", worst_c, cam_name(worst_c));
    // Per-frame for cam 0.
    for t in 0..16 {
        println!("  cam0 t={}: {:.3}", t, geometry::depth_absrel(&preds[t * N_CAMS].depth, &images[t * N_CAMS]));
    }
    for t in 0..16 {
        println!("  t={}: {:.3}", t, geometry::depth_absrel(&preds[t * N_CAMS + worst_c].depth, &images[t * N_CAMS + worst_c]));
    }
    // Error map for cam 4 at t=15.
    let t = 15;
    let img = &images[t * N_CAMS + worst_c];
    let pred = &preds[t * N_CAMS + worst_c];
    println!("error map cam {} t={} (X>1, *>0.4, +>0.15, - >0.05, .<0.05, s=sky):", worst_c, t);
    for r in (0..96).step_by(6) {
        let mut line = String::new();
        for cc in (0..128).step_by(2) {
            let p = r * 128 + cc;
            if img.sem[p] == 7 || img.depth[p] >= 399.0 { line.push('s'); continue; }
            let e = (pred.depth[p] - img.depth[p]).abs() / img.depth[p].max(0.5);
            let ch = if e < 0.05 { '.' } else if e < 0.15 { '-' } else if e < 0.4 { '+' } else if e < 1.0 { '*' } else { 'X' };
            line.push(ch);
        }
        println!("r{:2}: {}", r, line);
    }
}

fn cam_name(c: usize) -> &'static str {
    ["front", "left-front", "left-rear", "rear", "right-rear", "right-front"][c]
}
