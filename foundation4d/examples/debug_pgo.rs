use foundation4d::math::{Mat3, Rng, Se3, Vec3};
use foundation4d::pgo;

fn main() {
    // Straight-line trajectory with slight curve, 16 nodes.
    let n = 16;
    let mut gt = Vec::new();
    let mut x = 0.0;
    let mut yaw = 0.0;
    for _ in 0..n {
        gt.push(Se3 { r: Mat3::from_yaw(yaw), t: Vec3::new(x, 3.0 * (x / 40.0).sin(), 0.0) });
        x += 3.2;
        yaw += 0.01;
    }
    let mut rng = Rng::new(7);
    let windows = pgo::simulate_window_predictions(&gt, 4, 2, &mut rng);
    let factors = pgo::build_factors(&windows);
    println!("windows={} factors={}", windows.len(), factors.len());
    // Retained (raw) poses.
    let mut retained = vec![Se3::IDENTITY; n];
    for i in 0..n {
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
    let opt = pgo::optimize_pgo(&factors, n, 200);
    println!("RMSE retained = {:.4}", pgo::trajectory_rmse(&retained, &gt));
    println!("RMSE pgo      = {:.4}", pgo::trajectory_rmse(&opt, &gt));
    for i in 0..n {
        let yaw_gt = gt[i].r.log().z();
        let yaw_opt = opt[i].r.log().z();
        let yaw_ret = retained[i].r.log().z();
        println!("  yaw node {:2}: gt={:.4} retained={:.4} pgo={:.4}", i, yaw_gt, yaw_ret, yaw_opt);
        println!("  node {:2}: gt=({:6.2},{:6.2}) retained=({:6.2},{:6.2}) pgo=({:6.2},{:6.2})",
            i, gt[i].t.x(), gt[i].t.y(), retained[i].t.x(), retained[i].t.y(), opt[i].t.x(), opt[i].t.y());
    }
}
// rotation check appended
