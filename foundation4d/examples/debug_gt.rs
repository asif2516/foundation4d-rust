use foundation4d::raster;
use foundation4d::scene::{build_scene, CLASS_NAMES};

fn main() {
    let scene = build_scene(42);
    let mut images = Vec::new();
    raster::render_clip(&scene, &mut images);
    let img = &images[2 * 6 + 0]; // t=2, c=0 (front)
    println!("GT depth/sem along column 64, rows 70..95:");
    for r in 70..96 {
        let p = r * 128 + 64;
        println!("  r{:2}: depth={:7.2} sem={}", r, img.depth[p], CLASS_NAMES[img.sem[p] as usize]);
    }
    // Also check the ego camera pose and what's near.
    let pose = foundation4d::scene::cam_pose_world(&scene, 2, 0);
    println!("cam pose t = {:?}", pose.t.0);
    // Object closest to camera:
    for a in &scene.actors {
        let (p, yaw) = a.pose(0.8);
        println!("actor {} class {} pos {:?} dims {:?}", a.id, CLASS_NAMES[a.class as usize], p.0, a.dims.0);
    }
}
