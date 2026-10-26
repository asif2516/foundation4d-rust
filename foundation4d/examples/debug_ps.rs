use foundation4d::math::{Mat3, Se3, Vec3};
use foundation4d::scene::{CamImage, Tri, IMG_W};
use foundation4d::geometry::CameraModel;
use foundation4d::raster::render_camera;
use foundation4d::scene::build_scene;

fn main() {
    let scene = build_scene(3);
    let cam = CameraModel::new();
    let mut tris = Vec::new();
    for gy in 0..40 {
        for gx in 0..60 {
            let x0 = -30.0 + gx as f32;
            let y0 = -20.0 + gy as f32;
            let shade = ((gx * 7 + gy * 13) % 5) as f32 * 0.04;
            let col = [0.5 + shade, 0.5 + shade, 0.55 + shade];
            let x1 = x0 + 1.0;
            let y1 = y0 + 1.0;
            tris.push(Tri { a: Vec3::new(x0, y0, 10.0), b: Vec3::new(x1, y0, 10.0), c: Vec3::new(x1, y1, 10.0), color: col, sem: 3 });
            tris.push(Tri { a: Vec3::new(x0, y0, 10.0), b: Vec3::new(x1, y1, 10.0), c: Vec3::new(x0, y1, 10.0), color: col, sem: 3 });
        }
    }
    let pose_b = Se3 { r: Mat3::IDENTITY, t: Vec3::new(0.0, 1.0, 0.0) };
    let mut img_b = CamImage::default();
    render_camera(&scene, &tris, &pose_b, &cam.rays, cam.fx, cam.fy, cam.cx, cam.cy, &mut img_b, true);
    // No-noise render to isolate the base colors.
    let mut img_b2 = CamImage::default();
    render_camera(&scene, &tris, &pose_b, &cam.rays, cam.fx, cam.fy, cam.cx, cam.cy, &mut img_b2, false);
    for v in 38..50 {
        let q = v * IMG_W + 64;
        let y_wall = 1.0 + 10.0 * (v as f32 + 0.5 - 48.0) / cam.fx;
        println!("B(64,{}): base={} {} {} noisy={} {} {} wall y={:.3} quad={}", v,
            img_b2.rgb[q*3], img_b2.rgb[q*3+1], img_b2.rgb[q*3+2],
            img_b.rgb[q*3], img_b.rgb[q*3+1], img_b.rgb[q*3+2],
            y_wall, (y_wall + 20.0).floor());
    }
}
