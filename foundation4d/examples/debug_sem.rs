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
    // Pixel-level semantic confusion for t=8 all cams.
    let mut conf = vec![[0usize; 8]; 8];
    for c in 0..N_CAMS {
        let img = &images[8 * N_CAMS + c];
        let pred = &preds[8 * N_CAMS + c];
        for p in 0..foundation4d::scene::IMG_W * foundation4d::scene::IMG_H {
            conf[img.sem[p] as usize][pred.sem[p] as usize] += 1;
        }
    }
    println!("pixel-level sem confusion t=8 (gt -> pred):");
    for a in 0..8 {
        let tot: usize = conf[a].iter().sum();
        if tot > 20 {
            let row: Vec<String> = (0..8).map(|b| if conf[a][b] > 20 { format!("{}:{:.0}%", CLASS_NAMES[b], 100.0 * conf[a][b] as f32 / tot as f32) } else { String::new() }).collect();
            println!("  gt {:10} (n={:5}): {}", CLASS_NAMES[a], tot, row.join("  "));
        }
    }
}
