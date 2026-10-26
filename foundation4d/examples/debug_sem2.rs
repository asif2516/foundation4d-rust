use foundation4d::geometry::{self, CameraModel, PosePred};
use foundation4d::math::{Rng, Se3};
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
    // Front camera t=8: print pz stats for road pixels.
    let img = &images[8 * N_CAMS];
    let pred = &preds[8 * N_CAMS];
    let mut below = 0; let mut above = 0; let mut inband = 0;
    let mut sum_pz = 0.0; let mut n = 0;
    for p in 0..foundation4d::scene::IMG_W * foundation4d::scene::IMG_H {
        if img.sem[p] != 1 { continue; } // road
        let pz = pred.points[p * 3 + 2];
        sum_pz += pz; n += 1;
        if pz.abs() < 0.55 { inband += 1; } else if pz < 0.0 { below += 1; } else { above += 1; }
    }
    println!("road pixels n={} mean_pz={:.3} inband={} below={} above={}", n, sum_pz / n.max(1) as f32, inband, below, above);
    // Sample a few.
    for &(r, c) in &[(60usize, 64usize), (80usize, 64usize), (40usize, 64usize)] {
        let p = r * 128 + c;
        println!("  (r{},c{}) gt_sem={} pred_sem={} pz={:.3} depth={:.1} gt_depth={:.1}",
            r, c, CLASS_NAMES[img.sem[p] as usize], CLASS_NAMES[pred.sem[p] as usize],
            pred.points[p*3+2], pred.depth[p], img.depth[p]);
    }
}
