use foundation4d::math::{Mat3, Rng, Se3, Vec3};
use foundation4d::pgo;

fn main() {
    // Pure translation chain: nodes on a line, noisy relative measurements.
    let n = 16;
    let gt: Vec<Se3> = (0..n).map(|i| Se3 { r: Mat3::IDENTITY, t: Vec3::new(3.2 * i as f32, 0.2 * i as f32, 0.0) }).collect();
    let mut rng = Rng::new(11);
    // Direct relative measurements between ALL pairs within sep<=3 (like factors),
    // iid noise 0.05 on t_ij.
    let mut factors = Vec::new();
    for i in 0..n {
        for j in i + 1..(i + 4).min(n) {
            let z_true = Se3::between(&gt[i], &gt[j]);
            let z = Se3 {
                r: Mat3::IDENTITY,
                t: z_true.t.add(&Vec3::new(rng.normal_clamped(0.05), rng.normal_clamped(0.05), rng.normal_clamped(0.05))),
            };
            factors.push(pgo::Factor { i, j, z, weight: 1.0 });
        }
    }
    let opt = pgo::optimize_pgo(&factors, n, 10);
    println!("pure-translation: RMSE pgo = {:.4}", pgo::trajectory_rmse(&opt, &gt));
    // Expected ~ sigma/sqrt(effective) ~ 0.02-0.03. Print errors.
    for i in [1usize, 4, 8, 12, 15] {
        println!("  node {:2}: err = ({:.3},{:.3},{:.3})", i,
            opt[i].t.x() - gt[i].t.x(), opt[i].t.y() - gt[i].t.y(), opt[i].t.z() - gt[i].t.z());
    }
}
