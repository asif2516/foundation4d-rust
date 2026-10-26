use foundation4d::detection::detect_frame;
use foundation4d::geometry::{self, CameraModel, PosePred};
use foundation4d::math::{Rng, Se3};
use foundation4d::occupancy::{self};
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
    let pp = PosePred { ego: retained, rig: scene.cams_rig.to_vec() };
    let (preds, _) = geometry::predict_clip(&images, &pp, &cam, &mut Rng::new(1));
    let pose_pred = PosePred { ego: gt_ref.clone(), rig: scene.cams_rig.to_vec() };
    let bevs = occupancy::clip_bevs(&preds, &images, &pose_pred, &cam);
    let t = 8;
    let bev = &bevs[t];
    // Print ALL obstacle cells grouped loosely by position.
    let mut cells = Vec::new();
    for y in 0..120 {
        for x in 0..240 {
            let b = y * 240 + x;
            if bev.any_obstacle[b] == 1 {
                cells.push((x, y, bev.obst_height[b], bev.mean_rgb[b*3], bev.mean_rgb[b*3+1], bev.mean_rgb[b*3+2]));
            }
        }
    }
    println!("t={}: {} obstacle cells. Distribution by 10x10-cell blocks (world 5x5m):", t, cells.len());
    use std::collections::BTreeMap;
    let mut blocks: BTreeMap<(usize, usize), Vec<(f32, f32, f32, f32)>> = BTreeMap::new();
    for &(x, y, h, r, g, b) in &cells {
        blocks.entry((x / 10, y / 10)).or_default().push((h, r, g, b));
    }
    for ((bx, by), v) in blocks.iter() {
        let wx = -20.0 + (bx * 10) as f32 * 0.5;
        let wy = -30.0 + (by * 10) as f32 * 0.5;
        let mr: f32 = v.iter().map(|c| c.1).sum::<f32>() / v.len() as f32;
        let mg: f32 = v.iter().map(|c| c.2).sum::<f32>() / v.len() as f32;
        let mb: f32 = v.iter().map(|c| c.3).sum::<f32>() / v.len() as f32;
        let mh: f32 = v.iter().map(|c| c.0).sum::<f32>() / v.len() as f32;
        println!("  block ({:.0},{:.0})m: {} cells mean_h={:.1} rgb=({:.0},{:.0},{:.0})", wx, wy, v.len(), mh, mr, mg, mb);
    }
    let dets = detect_frame(bev, &mut Rng::new(1), false);
    println!("detections: {}", dets.len());
}
