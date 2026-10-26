use foundation4d::math::{Mat3, Rng, Se3, Vec3};
use foundation4d::pgo;

fn main() {
    // Zero-noise test: PGO must return GT exactly.
    let n = 16;
    let mut gt = Vec::new();
    let mut x = 0.0;
    let mut yaw = 0.0;
    for _ in 0..n {
        gt.push(Se3 { r: Mat3::from_yaw(yaw), t: Vec3::new(x, 3.0 * (x / 40.0).sin(), 0.0) });
        x += 3.2;
        yaw += 0.01;
    }
    // Zero-noise windows.
    let windows: Vec<pgo::WindowPred> = (0..5).map(|k| {
        let frames: Vec<usize> = (k * 3..k * 3 + 4).collect();
        let poses: Vec<Se3> = frames.iter().map(|&f| gt[f].clone()).collect();
        pgo::WindowPred { frames, poses }
    }).collect();
    let factors = pgo::build_factors(&windows);
    let opt = pgo::optimize_pgo(&factors, n, 100);
    println!("zero-noise RMSE = {:.6}", pgo::trajectory_rmse(&opt, &gt));
    for i in [4usize, 8, 12] {
        println!("  yaw {}: gt={:.4} pgo={:.4}", i, gt[i].r.log().z(), opt[i].r.log().z());
    }
    // Small iid rotation noise on measurements only.
    let mut rng = Rng::new(3);
    let windows2: Vec<pgo::WindowPred> = (0..5).map(|k| {
        let frames: Vec<usize> = (k * 3..k * 3 + 4).collect();
        let poses: Vec<Se3> = frames.iter().map(|&f| {
            let rr = Mat3::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), rng.normal_clamped(0.004));
            Se3 { r: gt[f].r.mul_mat(&rr), t: gt[f].t }
        }).collect();
        pgo::WindowPred { frames, poses }
    }).collect();
    let factors2 = pgo::build_factors(&windows2);
    let opt2 = pgo::optimize_pgo(&factors2, n, 100);
    let mut err = 0.0;
    for i in 0..n {
        err += (opt2[i].r.log().z() - gt[i].r.log().z()).abs();
    }
    println!("iid-rot-noise: mean yaw err = {:.4} rad", err / n as f32);
    for i in [4usize, 8, 12] {
        println!("  yaw {}: gt={:.4} pgo={:.4}", i, gt[i].r.log().z(), opt2[i].r.log().z());
    }
}
