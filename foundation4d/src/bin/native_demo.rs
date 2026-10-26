//! Native end-to-end demo: runs the full Foundation4D pipeline on the procedural
//! driving clip and prints per-stage statistics and quality metrics. Optionally
//! dumps PPM images for inspection.

use foundation4d::pipeline::Pipeline;

fn write_ppm(path: &str, w: usize, h: usize, data: &[u8]) {
    if let Ok(mut f) = std::fs::File::create(path) {
        use std::io::Write;
        let _ = write!(f, "P6\n{} {}\n255\n", w, h);
        // Strip the alpha channel (RGBA -> RGB).
        let mut rgb = Vec::with_capacity(w * h * 3);
        for px in data.chunks_exact(4) {
            rgb.extend_from_slice(&px[..3]);
        }
        let _ = f.write(&rgb);
    }
}

fn main() {
    let seed: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let out_dir = std::env::args().nth(2).unwrap_or_else(|| ".".to_string());

    let mut p = Pipeline::new(seed);
    let total = p.total_stages();
    println!("Foundation4D native demo — seed {seed}, {total} stages");
    let t_all = std::time::Instant::now();
    // Per-stage-class timing (native clock; the browser host does the same in JS).
    let nw_hint = 7usize;
    let mut timings = vec![0.0f64; 10]; // scene, render, windows, backbone, pgo, depth, occ, detect, track, gauss
    loop {
        let t0 = std::time::Instant::now();
        let s = p.step();
        let dt = t0.elapsed().as_secs_f64() * 1000.0;
        if s == usize::MAX {
            break;
        }
        let nw = nw_hint;
        let class = if s <= 2 {
            s
        } else if s < 3 + nw {
            3
        } else if s == 3 + nw {
            4
        } else if s < 4 + nw + 32 {
            5
        } else if s == 4 + nw + 32 {
            6
        } else if s == 5 + nw + 32 {
            7
        } else if s == 6 + nw + 32 {
            8
        } else {
            9
        };
        timings[class] += dt;
        println!(
            "  stage {:2}/{} done  ({:.0} ms cumulative)",
            s + 1,
            total,
            t_all.elapsed().as_millis()
        );
    }
    let names = ["scene", "render", "windows", "backbone", "pgo", "depth", "occupancy", "detect", "track", "gauss"];
    for (i, n) in names.iter().enumerate() {
        println!("  {n:10}: {:8.1} ms", timings[i]);
    }
    let s = &p.stats;
    println!("\n== Quality ==");
    println!("  depth AbsRel            : {:.4}", s[4]);
    println!("  depth AbsRel (far >30 m) : {:.4}", s[46]);
    println!("  occupancy IoU           : {:.3}", s[5]);
    println!("  occupancy sem accuracy  : {:.3}", s[6]);
    println!("  detections (raw, total) : {}", s[7]);
    println!("  tracks                  : {} (vehicles {}, persons {})", s[8], s[9], s[10]);
    println!("  trajectory RMSE raw     : {:.3} m", s[16]);
    println!("  trajectory RMSE stitch  : {:.3} m", s[17]);
    println!("  trajectory RMSE PGO     : {:.3} m", s[18]);
    println!("\n== Backbone (factorized attention) ==");
    println!("  total compute           : {:.2} GMAC ({:.2} GFLOP)", s[12], s[14]);
    println!("  forward time            : {:.0} ms -> {:.2} GFLOP/s", s[13], s[15]);
    println!("  d={}, layers={}, heads={}, tokens/img {} (pooled {})",
        s[37], s[38], s[39], s[40], s[41]);
    println!("  camera group tokens     : {}", s[42]);
    println!("  temporal group tokens   : {}", s[43]);
    println!("\n== Reconstruction ==");
    println!("  Gaussians               : {}", s[11]);
    println!("\n== Timings ==");
    println!("  scene {:.1} | render {:.1} | windows {:.1} | backbone {:.1} | depth {:.1} | pgo {:.1}",
        s[22], s[23], s[24], s[13], s[25], s[30]);
    println!("  occupancy {:.1} | detect {:.1} | track {:.1} | gauss {:.1} | TOTAL {:.1} ms",
        s[27], s[28], s[29], s[31], s[32]);

    // Dump a few images.
    let scene = p.scene.as_ref().unwrap();
    let (iw, ih) = (foundation4d::scene::IMG_W, foundation4d::scene::IMG_H);
    let f0 = &p.images[0 * 6 + 0];
    write_ppm(&format!("{out_dir}/cam_rgb.ppm"), iw, ih, &f0.rgb);
    // Depth map (predicted vs GT) as colormaps.
    let pred = &p.preds[0];
    let mut dimg = vec![0u8; iw * ih * 3];
    for i in 0..iw * ih {
        let d = (pred.depth[i] / 120.0).clamp(0.0, 1.0);
        dimg[i * 3] = (d * 255.0) as u8;
        dimg[i * 3 + 1] = (200.0 * (1.0 - d)) as u8;
        dimg[i * 3 + 2] = (255.0 * (1.0 - d)) as u8;
    }
    write_ppm(&format!("{out_dir}/cam_depth_pred.ppm"), iw, ih, &dimg);
    let _ = scene;
    println!("\nWrote {out_dir}/cam_rgb.ppm, {out_dir}/cam_depth_pred.ppm");

    // Gaussian spatial distribution sanity check.
    if let Some(g) = &p.gaussians {
        let mut min = [f32::MAX; 3];
        let mut max = [f32::MIN; 3];
        let mut mean = [0.0f32; 3];
        let mut n_static = 0usize;
        let mut n_actor = 0usize;
        for i in 0..g.n {
            for k in 0..3 {
                let v = g.pos[i * 3 + k];
                min[k] = min[k].min(v);
                max[k] = max[k].max(v);
                mean[k] += v;
            }
            if g.actor[i] < 0 { n_static += 1; } else { n_actor += 1; }
        }
        println!("Gaussian positions: min {:?} max {:?} mean {:?} (static {} / actor {})",
            min, max, [mean[0] / g.n as f32, mean[1] / g.n as f32, mean[2] / g.n as f32], n_static, n_actor);
        // Histogram: lateral (y) distribution of static Gaussians near x in [-15, 15].
        let mut hist = vec![0usize; 16];
        for i in 0..g.n {
            if g.actor[i] >= 0 { continue; }
            let x = g.pos[i * 3];
            if x < -15.0 || x > 15.0 { continue; }
            let y = g.pos[i * 3 + 1];
            let b = (((y + 40.0) / 5.0) as usize).min(15);
            hist[b] += 1;
        }
        println!("y-histogram (x in [-15,15]), 5m bins from -40m: {:?}", hist);
        // Per-source-camera counts.
        let mut per_cam = vec![0usize; 6];
        let _ = &mut per_cam;
    }

    // Render a novel view (browser-equivalent camera for debugging).
    let eye = [12.0f32, -14.0, 9.0];
    let fwd = [14.0f32, 8.0, -8.0];
    p.set_render_size(320, 192);
    p.render_view(eye, fwd, 1.0, 3.2, 0);
    #[cfg(feature = "splat_dbg")]
    println!("dbg: projected={} mean_ru={:.1}px near(<10m)={} entries={} blends={}",
        p.renderer.dbg_proj, p.renderer.dbg_med_ru, p.renderer.dbg_near, p.renderer.drawn, p.renderer.blends);
    write_ppm(&format!("{out_dir}/recon_view2.ppm"), 320, 192, &p.renderer.rgba);
    #[cfg(feature = "splat_dbg")]
    {
        let ws = &p.renderer.dbg_wsum;
        let mut mx = 0.0f32;
        let mut over1 = 0usize;
        for &v in ws.iter() {
            mx = mx.max(v);
            if v > 1.05 { over1 += 1; }
        }
        println!("dbg: max weight-sum = {:.3}, pixels with wsum>1.05: {} / {}", mx, over1, ws.len());
    }
    println!("browser-cam view written");
    let r = &p.renderer;
    write_ppm(&format!("{out_dir}/recon_view.ppm"), r.w, r.h, &r.rgba);
    println!("Rendered recon view: {}x{} in {:.1} ms ({} tile entries, {:.2}M blends)",
        r.w, r.h, r.last_ms, r.drawn, r.blends as f64 / 1e6);
}
