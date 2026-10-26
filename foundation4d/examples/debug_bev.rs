use foundation4d::geometry::{self, CameraModel, PosePred};
use foundation4d::math::{Rng, Se3};
use foundation4d::occupancy::{self, NX, NY, VOXEL, X_MIN, Y_MIN};
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
    let t = 8;
    let bev = &bevs[t];
    // GT vehicle at (21.2, 6.0): dump obstacle cells within 6m of it.
    let (gx, gy) = (13.0f32, 6.7f32);
    println!("obstacle cells within 7m of GT vehicle (21.2, 6.0) at t=4:");
    let mut count = 0;
    for y in 0..NY {
        for x in 0..NX {
            let b = y * NX + x;
            if bev.any_obstacle[b] == 0 { continue; }
            let wx = X_MIN + x as f32 * VOXEL;
            let wy = Y_MIN + y as f32 * VOXEL;
            let d = ((wx - gx).powi(2) + (wy - gy).powi(2)).sqrt();
            if d < 5.0 {
                count += 1;
                if count < 40 {
                    println!("  cell ({:.1},{:.1}) obst_h={:.1} cls={} rgb={:.0},{:.0},{:.0}",
                        wx, wy, bev.obst_height[b], CLASS_NAMES[bev.cls[b] as usize],
                        bev.mean_rgb[b*3], bev.mean_rgb[b*3+1], bev.mean_rgb[b*3+2]);
                }
            }
        }
    }
    println!("total near-vehicle obstacle cells: {}", count);
    // Print all GT actors in BEV cell coords for reference.
    println!("GT vehicles within 40m:");
    for a in &scene.actors {
        let (p, _) = a.pose(t as f32 * DT);
        let rel = p.sub(&gt_ref[t].t);
        if rel.x() > -5.0 && rel.x() < 40.0 && rel.y().abs() < 30.0 {
            println!("  {} at ({:.1},{:.1})", CLASS_NAMES[a.class as usize], rel.x(), rel.y());
        }
    }
}
