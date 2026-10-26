use foundation4d::math::{Mat3, Rng, Se3, Vec3};
use foundation4d::pgo;

fn factor_residual(factors: &[pgo::Factor], poses: &[Se3]) -> f32 {
    let mut s = 0.0;
    for f in factors {
        let e = Se3::between(&poses[f.i], &poses[f.j]);
        let dr = e.r.mul_mat(&f.z.r.transpose());
        let dt = e.t.sub(&f.z.t);
        s += f.weight * (dr.frob_norm2() * 0.25 + dt.norm2());
    }
    s
}

fn main() {
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
    let opt = pgo::optimize_pgo(&factors, n, 60);
    // Residual comparison: GT vs PGO solution.
    let res_gt = factor_residual(&factors, &gt);
    let res_pgo = factor_residual(&factors, &opt);
    println!("factor residual GT  = {:.5}", res_gt);
    println!("factor residual PGO = {:.5}", res_pgo);
    // Retained for reference.
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
    println!("factor residual retained = {:.5}", factor_residual(&factors, &retained));
    println!("RMSE retained = {:.4}", pgo::trajectory_rmse(&retained, &gt));
    println!("RMSE pgo      = {:.4}", pgo::trajectory_rmse(&opt, &gt));
}
