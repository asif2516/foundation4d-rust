use foundation4d::detection;
use foundation4d::geometry::{self, CameraModel, PosePred};
use foundation4d::math::{Rng, Se3};
use foundation4d::occupancy;
use foundation4d::pgo;
use foundation4d::raster;
use foundation4d::scene::{build_scene, CLASS_NAMES, N_CAMS, DT};

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
    let pp = PosePred { ego: retained, rig: scene.cams_rig.to_vec() };
    let (preds, _) = geometry::predict_clip(&images, &pp, &cam, &mut Rng::new(1));
    let pose_pred = PosePred { ego: gt_ref.clone(), rig: scene.cams_rig.to_vec() };
    let bevs = occupancy::clip_bevs(&preds, &images, &pose_pred, &cam);
    for t in [4usize, 8usize, 12usize] {
        let bev = &bevs[t];
        let n_obs = bev.any_obstacle.iter().filter(|&&v| v == 1).count();
        let mut det = detection::detect_frame(bev, &mut Rng::new(1), false);
        println!("t={}: obstacle cells={} raw_detections={}", t, n_obs, det.len());
        det.sort_by(|a, b| a.center[0].partial_cmp(&b.center[0]).unwrap());
        for d in det.iter() {
            println!("   {} at ({:.1},{:.1}) dims ({:.1},{:.1},{:.1}) yaw {:.0}deg",
                CLASS_NAMES[d.class as usize], d.center[0], d.center[1], d.dims[0], d.dims[1], d.dims[2], d.yaw.to_degrees());
        }
        println!("   GT actors (within 70m, |y|<35, clip frame):");
        for a in &scene.actors {
            let (p, yaw) = a.pose(t as f32 * DT);
            // Full clip-frame transform of the vehicle position relative to the ego.
            let rel_world = p.sub(&scene.ego_gt[t].t);
            let rel_clip = scene.ego_gt[0].r.transpose().mul_vec(&rel_world);
            if rel_clip.x() > -10.0 && rel_clip.x() < 80.0 && rel_clip.y().abs() < 35.0 {
                println!("   GT {} at ({:.1},{:.1}) yaw {:.0}deg", CLASS_NAMES[a.class as usize], rel_clip.x(), rel_clip.y(), yaw.to_degrees());
            }
        }
    }
}
