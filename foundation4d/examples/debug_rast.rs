use foundation4d::math::{Mat3, Se3, Vec3};
use foundation4d::raster::{render_camera, precompute_rays};
use foundation4d::scene::{build_scene, CamImage, Tri};

fn main() {
    // One 8x8m ground quad at z=0, subdivided 1m and 4m — compare depth accuracy.
    let scene = build_scene(1);
    let fx = (128.0 * 0.5) / (80.0_f32.to_radians() * 0.5).tan();
    let rays = precompute_rays(fx, fx, 64.0, 48.0);
    // Camera: pitch -7 deg, height 1.45, at origin looking +x.
    let cam_basis = |yaw: f32, pitch: f32| -> Mat3 {
        let x_cam = Vec3::new(yaw.sin(), -yaw.cos(), 0.0);
        let y_cam = Vec3::new(0.0, 0.0, -1.0);
        let z_cam = Vec3::new(yaw.cos(), yaw.sin(), 0.0);
        let rx = Mat3([x_cam.x(), y_cam.x(), z_cam.x(), x_cam.y(), y_cam.y(), z_cam.y(), x_cam.z(), y_cam.z(), z_cam.z()]);
        rx.mul_mat(&Mat3::from_pitch(pitch))
    };
    let pose = Se3 { r: cam_basis(0.0, -7.0_f32.to_radians()), t: Vec3::new(0.0, 0.0, 1.45) };
    for step in [1.0f32, 4.0] {
        let mut tris = Vec::new();
        for gx in 0..(8.0 / step) as usize {
            for gy in 0..(8.0 / step) as usize {
                let x0 = gx as f32 * step;
                let y0 = gy as f32 * step - 4.0;
                let x1 = x0 + step;
                let y1 = y0 + step;
                tris.push(Tri { a: Vec3::new(x0, y0, 0.0), b: Vec3::new(x1, y0, 0.0), c: Vec3::new(x1, y1, 0.0), color: [0.5, 0.5, 0.5], sem: 4 });
                tris.push(Tri { a: Vec3::new(x0, y0, 0.0), b: Vec3::new(x1, y1, 0.0), c: Vec3::new(x0, y1, 0.0), color: [0.5, 0.5, 0.5], sem: 4 });
            }
        }
        let mut img = CamImage::default();
        render_camera(&scene, &tris, &pose, &rays, fx, fx, 64.0, 48.0, &mut img, false);
        // Check row 80, col 64: analytic depth.
        let p = 80 * 128 + 64;
        let v_ang = ((80.5 - 48.0) / fx).atan(); // below camera axis
        let below_horiz = v_ang + 7.0_f32.to_radians();
        let x_horiz = 1.45 / below_horiz.tan();
        let z_analytic = x_horiz * (7.0_f32.to_radians()).cos() + 1.45 * (7.0_f32.to_radians()).sin();
        println!("step={}m: r80c64 raster={:.3} analytic={:.3}", step, img.depth[p], z_analytic);
        // Check a few more rows.
        for r in [70usize, 75, 85, 90] {
            let p = r * 128 + 64;
            let va = ((r as f32 + 0.5 - 48.0) / fx).atan();
            let bh = va + 7.0_f32.to_radians();
            let xh = 1.45 / bh.tan();
            let za = xh * (7.0_f32.to_radians()).cos() + 1.45 * (7.0_f32.to_radians()).sin();
            println!("   r{}: raster={:.3} analytic={:.3} err={:.3}", r, img.depth[p], za, img.depth[p] - za);
        }
    }
}
