//! Debug: per-frame, per-camera absrel over the whole clip (GT poses).

use foundation4d::geometry::{self, CameraModel, DepthScratch, PosePred};
use foundation4d::raster;
use foundation4d::scene::{build_scene, N_CAMS, N_FRAMES, SEM_SKY, IMG_H, IMG_W};
use foundation4d::math::Se3;

fn main() {
    let seed: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let scene = build_scene(seed as u64);
    let cam = CameraModel::new();
    let mut images = Vec::new();
    raster::render_clip(&scene, &mut images);
    let gt_ref: Vec<Se3> = scene.ego_gt.iter().map(|p| Se3::between(&scene.ego_gt[0], p)).collect();
    let poses = PosePred { ego: gt_ref, rig: scene.cams_rig.to_vec() };
    let mut scratch = DepthScratch::new();
    let npix = IMG_W * IMG_H;

    let mut total = 0.0f64;
    let mut total_far = 0.0f64;
    let mut n_tot = 0usize;
    let mut n_far = 0usize;
    let mut total_sf = 0.0f64;
    let mut n_sf = 0usize;
    let mut worst = vec![(0.0f32, 0usize); N_FRAMES * N_CAMS + 1];
    for t in 0..N_FRAMES {
        let use_sgm = std::env::var("F4D_NO_SGM").map(|v| v != "1").unwrap_or(true);
        let preds = geometry::predict_frame_mode(&images, &poses, &cam, &mut scratch, t, use_sgm);
        for c in 0..N_CAMS {
            let gt = &images[t * N_CAMS + c];
            let mut acc = 0.0;
            let mut n = 0;
            let mut facc = 0.0;
            let mut fn_ = 0;
            let mut sfacc = 0.0;
            let mut sfn = 0;
            for p in 0..npix {
                if gt.sem[p] == SEM_SKY || gt.depth[p] >= 400.0 { continue; }
                let e = (preds[c].depth[p] - gt.depth[p]).abs() / gt.depth[p].max(0.5);
                acc += e; n += 1;
                let is_actor = gt.sem[p] == foundation4d::scene::SEM_VEHICLE || gt.sem[p] == foundation4d::scene::SEM_PERSON;
                if gt.depth[p] > 30.0 {
                    facc += e; fn_ += 1;
                    if !is_actor { sfacc += e; sfn += 1; }
                }
            }
            total_sf += sfacc as f64; n_sf += sfn;
            let a = acc / n.max(1) as f32;
            let f = facc / fn_.max(1) as f32;
            total += acc as f64; n_tot += n;
            total_far += facc as f64; n_far += fn_;
            worst[t * N_CAMS + c] = (a, t * N_CAMS + c);
            print!("t{t}c{c}:{a:.2} ");
        }
        println!();
    }
    println!("CLIP absrel {:.4} | far {:.4} | far-static {:.4}", total / n_tot as f64, total_far / n_far as f64, total_sf / n_sf.max(1) as f64);
    worst.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    print!("worst views: ");
    for (a, i) in worst.iter().take(12) {
        print!("t{}c{}:{:.2} ", i / N_CAMS, i % N_CAMS, a);
    }
    println!();
}
