use foundation4d::math::{Mat3, Rng, Se3, Vec3};

fn main() {
    // Test the sim3 path directly with debug prints.
    let r = Mat3::from_yaw(0.4);
    let t = Vec3::new(1.0, -2.0, 0.5);
    let xs: Vec<Vec3> = (0..20)
        .map(|i| Vec3::new(i as f32 * 0.7, (i % 5) as f32 * 1.3, (i % 3) as f32 * 0.8))
        .collect();
    let ys: Vec<Vec3> = xs.iter().map(|x| r.mul_vec(x).add(&t)).collect();
    let est = foundation4d::pgo::sim3_irls(&xs, &ys, 10);
    println!("est R yaw = {:?}", est.r.log().z());
    println!("est t = {:?}", est.t.0);
    for (x, y) in xs.iter().zip(ys.iter()).take(5) {
        let p = est.apply(x);
        println!("err = {:.4}", p.sub(y).norm());
    }
    // Check covariance convention with a direct horn test via public path
    let mut rng = Rng::new(3);
    let _ = &mut rng;
}
