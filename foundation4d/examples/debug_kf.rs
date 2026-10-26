use foundation4d::detection::Cuboid;
use foundation4d::tracking::{track_detections, Kf};
use foundation4d::math::Rng;
use foundation4d::scene::{DT, N_FRAMES, SEM_VEHICLE};

fn main() {
    // Direct KF test: moving object, exact velocity measurements.
    let mut kf = Kf::new([0.0, 1.0], [5.0, 0.0]);
    for t in 0..N_FRAMES {
        kf.predict(DT);
        let x_true = 5.0 * (t as f32 * DT);
        kf.update(&[x_true, 1.0, 5.0, 0.0]);
        println!("t={} kf.x = {:.3} {:.3} {:.3} {:.3}", t, kf.x[0], kf.x[1], kf.x[2], kf.x[3]);
    }
    println!("---tracking---");
    let mut rng = Rng::new(11);
    let dets: Vec<Vec<Cuboid>> = (0..N_FRAMES)
        .map(|t| {
            let x = 5.0 * (t as f32 * DT) + rng.normal_clamped(0.12);
            vec![Cuboid { center: [x, 1.0 + rng.normal_clamped(0.12), 0.8], dims: [4.4, 1.8, 1.5], yaw: 0.0, vel: [5.0, 0.0], class: SEM_VEHICLE, score: 0.9 }]
        })
        .collect();
    let tracks = track_detections(&dets, &mut rng);
    println!("n_tracks = {}", tracks.len());
    for tr in &tracks {
        println!("track {} class {} n={}", tr.id, tr.class, tr.states.len());
        for s in &tr.states {
            println!("  t={:.1} x={:.2} y={:.2} vx={:.2} vy={:.2} yaw={:.2}", s[0], s[1], s[2], s[3], s[4], s[5]);
        }
    }
}
