use foundation4d::math::{Mat3, Rng, Se3, Vec3};
use foundation4d::pgo;
const WSIZE: usize = 8;
const WSTRIDE: usize = 4;

fn main() {
    let n = 16;
    let mut gt = Vec::new();
    let mut x = 0.0; let mut yaw = 0.0;
    for _ in 0..n {
        gt.push(Se3 { r: Mat3::from_yaw(yaw), t: Vec3::new(x, 3.0 * (x / 40.0).sin(), 0.0) });
        x += 3.2; yaw += 0.01;
    }
    for &(sr, st, dr, dt) in &[(0.001f32, 0.012f32, 0.002f32, 0.04f32), (0.001, 0.012, 0.0025, 0.05), (0.0015, 0.015, 0.002, 0.04)] {
        let mut rng = Rng::new(7);
        let mut windows = Vec::new();
        let mut s = 0usize;
        while s + WSIZE <= n {
            let frames: Vec<usize> = (s..s + 4).collect();
            let drift_yaw = rng.normal_clamped(dr);
            let drift_t = Vec3::new(rng.normal_clamped(dt), rng.normal_clamped(dt), 0.0);
            let anchor_inv = gt[s].inverse();
            let poses: Vec<Se3> = frames.iter().map(|&i| {
                let k = (i - s) as f32;
                if k == 0.0 { return Se3::IDENTITY; }
                let axis = Vec3::new(rng.normal_clamped(1.0), rng.normal_clamped(1.0), rng.normal_clamped(1.0)).normalized();
                let rr = Mat3::from_axis_angle(&axis, rng.normal_clamped(sr + 0.001 * k) + k * drift_yaw);
                let tt = Vec3::new(rng.normal_clamped(st + 0.012 * k) + k * drift_t.x(), rng.normal_clamped(st + 0.012 * k) + k * drift_t.y(), 0.0);
                anchor_inv.compose(&gt[i]).compose(&Se3 { r: rr, t: tt })
            }).collect();
            windows.push(pgo::WindowPred { frames, poses });
            s += WSTRIDE;
        }
        let factors = pgo::build_factors(&windows);
        let opt = pgo::optimize_pgo(&factors, n, 40);
        let stitched_windows = pgo::stitch_sequential(&windows);
        let mut stitched = vec![Se3::IDENTITY; n];
        for i in 0..n {
            'o: for (k, w) in windows.iter().enumerate() {
                if let Some(pos) = w.frames.iter().position(|&f| f == i) { stitched[i] = stitched_windows[k][pos]; break 'o; }
            }
        }
        println!("sr={} st={} drift=({},{}) -> RMSE stitch={:.3} pgo={:.3}",
            sr, st, dr, dt, pgo::trajectory_rmse(&stitched, &gt), pgo::trajectory_rmse(&opt, &gt));
    }
}
