use foundation4d::geometry::{cam_pose, CameraModel, PosePred};
use foundation4d::math::Se3;
use foundation4d::scene::{build_scene, N_CAMS};

fn main() {
    let scene = build_scene(42);
    let cam = CameraModel::new();
    let pp = PosePred {
        ego: (0..16).map(|t| Se3::between(&scene.ego_gt[0], &scene.ego_gt[t])).collect(),
        rig: scene.cams_rig.to_vec(),
    };
    let pose = cam_pose(&pp, 15, 4);
    println!("cam4 t=15 pose: t = {:?}", pose.t.0);
    println!("  R cols (x,y,z axes): x={:?} y={:?} z={:?}",
        [pose.r.0[0], pose.r.0[3], pose.r.0[6]],
        [pose.r.0[1], pose.r.0[4], pose.r.0[7]],
        [pose.r.0[2], pose.r.0[5], pose.r.0[8]]);
    // GT scene camera pose for comparison:
    let gt = foundation4d::scene::cam_pose_world(&scene, 15, 4);
    let gt_clip = Se3::between(&scene.ego_gt[0], &gt);
    println!("GT cam4 t=15 (clip frame): t = {:?}", gt_clip.t.0);
    let p = 80 * 128 + 30;
    let ray = &cam.rays[p];
    println!("ray(r80,c30) = {:?}", ray.0);
    let gt_depth = 2.63f32;
    let s = gt_depth / ray.z();
    let pw = pose.apply(&ray.scale(s));
    println!("point at gt depth via pp pose: {:?}", pw.0);
    let pw2 = gt_clip.apply(&ray.scale(s));
    println!("point at gt depth via GT pose:  {:?}", pw2.0);
}
