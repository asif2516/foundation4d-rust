use foundation4d::geometry::{cam_pose, CameraModel, PosePred, sample_rgb_pub};
use foundation4d::math::Se3;
use foundation4d::raster;
use foundation4d::scene::{build_scene, N_CAMS};

fn main() {
    let scene = build_scene(42);
    let cam = CameraModel::new();
    let mut images = Vec::new();
    raster::render_clip(&scene, &mut images);
    let pp = PosePred {
        ego: (0..16).map(|t| Se3::between(&scene.ego_gt[0], &scene.ego_gt[t])).collect(),
        rig: scene.cams_rig.to_vec(),
    };
    let (t, c) = (15usize, 4usize);
    let img_ref = &images[t * N_CAMS + c];
    let pose_ref = cam_pose(&pp, t, c);
    let p = 80 * 128 + 30; // bottom-left (cam3 overlap side)
    let gt = img_ref.depth[p];
    let ray = &cam.rays[p];
    let col = [img_ref.rgb[p*3] as f32, img_ref.rgb[p*3+1] as f32, img_ref.rgb[p*3+2] as f32];
    println!("cam4 t=15 pixel (r80,c30) gt={:.2} rgb={:?}", gt, col);
    let mut sources: Vec<(usize, usize)> = vec![(t, 3), (t, 5)];
    for &dt in &[-2i32, -1, 1, 2] {
        let tt = t as i32 + dt;
        if tt >= 0 && (tt as usize) < 16 { sources.push((tt as usize, c)); }
    }
    for k in 0..40 {
        let d = 1.2f32 * (65.0f32/1.2).powf(k as f32 / 39.0);
        let s = d / ray.z();
        let pw = pose_ref.apply(&ray.scale(s));
        let mut line = format!("  d={:6.2} w=({:5.1},{:5.1}):", d, pw.x(), pw.y());
        for &(tt, sc) in &sources {
            let pose_s = cam_pose(&pp, tt, sc);
            let pc = pose_s.inverse().apply(&pw);
            let cost = if pc.z() <= 0.1 { "  MISS".to_string() } else {
                let u = cam.fx * pc.x() / pc.z() + cam.cx - 0.5;
                let v = cam.fy * pc.y() / pc.z() + cam.cy - 0.5;
                match sample_rgb_pub(&images[tt * N_CAMS + sc].rgb, u, v) {
                    Some(c2) => format!("  {:5.0}", (col[0]-c2[0]).abs() + (col[1]-c2[1]).abs() + (col[2]-c2[2]).abs()),
                    None => "  MISS".to_string(),
                }
            };
            line.push_str(&cost);
        }
        println!("{}", line);
    }
}
