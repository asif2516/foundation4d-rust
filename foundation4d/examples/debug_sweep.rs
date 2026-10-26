//! Debug: coarse-to-fine sweep behavior on one real camera frame vs GT.
//! Dumps per-stage diagnostics to locate depth regressions.

use foundation4d::geometry::{self, CameraModel, DepthScratch, SweepSources, D_MAX, D_MIN};
use foundation4d::math::{Se3, Vec3};
use foundation4d::raster;
use foundation4d::scene::{build_scene, N_CAMS, N_FRAMES, IMG_H, IMG_W};

fn main() {
    let seed: u32 = std::env::args().nth(1).and_then(|s| s.parse().ok()).unwrap_or(42);
    let scene = build_scene(seed as u64);
    let cam = CameraModel::new();
    let mut images = Vec::new();
    raster::render_clip(&scene, &mut images);

    // GT poses (clip frame): reference camera trajectory.
    let gt_ref: Vec<Se3> = scene.ego_gt.iter().map(|p| Se3::between(&scene.ego_gt[0], p)).collect();
    let poses = geometry::PosePred { ego: gt_ref.clone(), rig: scene.cams_rig.to_vec() };

    // Pick a mid-clip frame and the front camera (or from argv).
    let t: usize = std::env::args().nth(2).and_then(|s| s.parse().ok()).unwrap_or(8);
    let c: usize = std::env::args().nth(3).and_then(|s| s.parse().ok()).unwrap_or(0);
    let idx = t * N_CAMS + c;
    let pose = geometry::cam_pose(&poses, t, c);
    let src = {
        let mut imgs: Vec<&[u8]> = Vec::new();
        let mut ps: Vec<Se3> = Vec::new();
        let mut w: Vec<f32> = Vec::new();
        let prev_c = (c + N_CAMS - 1) % N_CAMS;
        let next_c = (c + 1) % N_CAMS;
        imgs.push(&images[t * N_CAMS + prev_c].rgb);
        ps.push(geometry::cam_pose(&poses, t, prev_c));
        w.push(1.6);
        imgs.push(&images[t * N_CAMS + next_c].rgb);
        ps.push(geometry::cam_pose(&poses, t, next_c));
        w.push(1.6);
        for &dt in &[-2i32, -1, 1, 2] {
            let tt = t as i32 + dt;
            if tt >= 0 && (tt as usize) < N_FRAMES {
                imgs.push(&images[tt as usize * N_CAMS + c].rgb);
                ps.push(geometry::cam_pose(&poses, tt as usize, c));
                w.push(if dt.abs() == 2 { 0.6 } else { 1.0 });
            }
        }
        SweepSources { imgs, poses: ps, weights: w }
    };

    let gt = &images[idx];
    let npix = IMG_W * IMG_H;

    // ---- 0) OLD algorithm (single-scale plane sweep, WTA + box smoothing) ----
    {
        const N_BINS: usize = 40;
        const D_MINO: f32 = 1.2;
        const D_MAXO: f32 = 65.0;
        let npix = IMG_W * IMG_H;
        let n_src = src.imgs.len();
        let w_tot: f32 = src.weights.iter().sum();
        let mut cost = vec![0.0f32; npix * N_BINS];
        let mut bins = [0.0f32; N_BINS];
        for k in 0..N_BINS {
            bins[k] = D_MINO * (D_MAXO / D_MINO).powf(k as f32 / (N_BINS - 1) as f32);
        }
        for p in 0..npix {
            let ray = &cam.rays[p];
            let r = &pose.r;
            let ro = &pose.t;
            let (rx, ry, rz) = (ray.x(), ray.y(), ray.z());
            let wx = r.0[0] * rx + r.0[1] * ry + r.0[2] * rz;
            let wy = r.0[3] * rx + r.0[4] * ry + r.0[5] * rz;
            let wz = r.0[6] * rx + r.0[7] * ry + r.0[8] * rz;
            let col = [gt.rgb[p * 3] as f32, gt.rgb[p * 3 + 1] as f32, gt.rgb[p * 3 + 2] as f32];
            for k in 0..N_BINS {
                let s = bins[k] / rz;
                let px = ro.x() + wx * s;
                let py = ro.y() + wy * s;
                let pz = ro.z() + wz * s;
                let mut cacc = 0.0f32;
                let mut wsum = 0.0f32;
                for (si, sidx) in src.poses.iter().enumerate() {
                    let sx = sidx.r.0[0] * px + sidx.r.0[3] * py + sidx.r.0[6] * pz - sidx.r.0[0] * sidx.t.x() - sidx.r.0[3] * sidx.t.y() - sidx.r.0[6] * sidx.t.z();
                    let sy = sidx.r.0[1] * px + sidx.r.0[4] * py + sidx.r.0[7] * pz - sidx.r.0[1] * sidx.t.x() - sidx.r.0[4] * sidx.t.y() - sidx.r.0[7] * sidx.t.z();
                    let sz = sidx.r.0[2] * px + sidx.r.0[5] * py + sidx.r.0[8] * pz - sidx.r.0[2] * sidx.t.x() - sidx.r.0[5] * sidx.t.y() - sidx.r.0[8] * sidx.t.z();
                    if sz < 0.1 { continue; }
                    let invz = 1.0 / sz;
                    let u = cam.fx * sx * invz + cam.cx - 0.5;
                    let v = cam.fy * sy * invz + cam.cy - 0.5;
                    if let Some(c2) = foundation4d::geometry::sample_rgb_pub(src.imgs[si], u, v) {
                        cacc += src.weights[si] * ((col[0] - c2[0]).abs() + (col[1] - c2[1]).abs() + (col[2] - c2[2]).abs());
                        wsum += src.weights[si];
                    }
                }
                if wsum <= 0.0 {
                    cost[p * N_BINS + k] = 120.0;
                } else {
                    cost[p * N_BINS + k] = cacc / wsum + 6.0 * (w_tot - wsum) / w_tot.max(1.0) * 3.0;
                }
            }
        }
        // 3-wide box smoothing.
        let mut tmp = vec![0.0f32; npix * N_BINS];
        for y in 0..IMG_H {
            for x in 0..IMG_W {
                for k in 0..N_BINS {
                    let mut s = 0.0;
                    let mut n = 0;
                    for dx in -1i32..=1 {
                        let xx = x as i32 + dx;
                        if xx >= 0 && xx < IMG_W as i32 {
                            s += cost[(y * IMG_W + xx as usize) * N_BINS + k];
                            n += 1;
                        }
                    }
                    tmp[(y * IMG_W + x) * N_BINS + k] = s / n as f32;
                }
            }
        }
        for y in 0..IMG_H {
            for x in 0..IMG_W {
                for k in 0..N_BINS {
                    let mut s = 0.0;
                    let mut n = 0;
                    for dy in -1i32..=1 {
                        let yy = y as i32 + dy;
                        if yy >= 0 && yy < IMG_H as i32 {
                            s += tmp[(yy as usize * IMG_W + x) * N_BINS + k];
                            n += 1;
                        }
                    }
                    cost[(y * IMG_W + x) * N_BINS + k] = s / n as f32;
                }
            }
        }
        // WTA + parabolic.
        let mut od = vec![400.0f32; npix];
        for p in 0..npix {
            let row = &cost[p * N_BINS..(p + 1) * N_BINS];
            let mut best = 0usize;
            let mut best_c = f32::INFINITY;
            for k in 0..N_BINS {
                if row[k] < best_c { best_c = row[k]; best = k; }
            }
            let t = if best > 0 && best < N_BINS - 1 {
                let c0 = row[best - 1];
                let c1 = row[best];
                let c2 = row[best + 1];
                let denom = c0 - 2.0 * c1 + c2;
                if denom > 1e-6 { ((c0 - c2) / (2.0 * denom)).clamp(-0.5, 0.5) } else { 0.0 }
            } else { 0.0 };
            let log_d = bins[best].ln() + t * ((D_MAXO / D_MINO).ln() / (N_BINS - 1) as f32);
            od[p] = log_d.exp();
        }
        // Metrics: full + static-only + actor-only.
        let mut acc = [0.0f64; 3];
        let mut cnt = [0usize; 3];
        for p in 0..npix {
            if gt.sem[p] == foundation4d::scene::SEM_SKY || gt.depth[p] >= 400.0 { continue; }
            let is_actor = gt.sem[p] == foundation4d::scene::SEM_VEHICLE || gt.sem[p] == foundation4d::scene::SEM_PERSON;
            let e = (od[p] - gt.depth[p]).abs() / gt.depth[p].max(0.5);
            acc[0] += e as f64; cnt[0] += 1;
            acc[1 + is_actor as usize] += e as f64; cnt[1 + is_actor as usize] += 1;
        }
        println!("OLD-ALGO on this frame: absrel {:.4} | static {:.4} (n={}) | actor {:.4} (n={})",
            acc[0] / cnt[0] as f64, acc[1] / cnt[1].max(1) as f64, cnt[1], acc[2] / cnt[2].max(1) as f64, cnt[2]);
        // Old-algo at a few row-72 pixels.
        let mut line = String::new();
        for x in (0..IMG_W).step_by(16) {
            let p = 72 * IMG_W + x;
            if gt.sem[p] == foundation4d::scene::SEM_SKY { continue; }
            line.push_str(&format!("({}: gt {:.1} old {:.1}) ", x, gt.depth[p], od[p]));
        }
        println!("OLD row 72: {line}");
    }

    // ---- 1) sanity: fast transform vs explicit world-point transform ----
    {
        let sfr = Se3::between(&src.poses[0], &pose);
        let p = 1000usize; // some pixel
        let ray = &cam.rays[p];
        let d = 20.0f32;
        let s = d / ray.z();
        let fast = Vec3::new(
            sfr.r.0[0] * ray.x() * s + sfr.r.0[1] * ray.y() * s + sfr.r.0[2] * ray.z() * s + sfr.t.x(),
            sfr.r.0[3] * ray.x() * s + sfr.r.0[4] * ray.y() * s + sfr.r.0[5] * ray.z() * s + sfr.t.y(),
            sfr.r.0[6] * ray.x() * s + sfr.r.0[7] * ray.y() * s + sfr.r.0[8] * ray.z() * s + sfr.t.z(),
        );
        // Explicit: world point then into source frame.
        let pw = pose.apply(&Vec3::new(ray.x() * s, ray.y() * s, ray.z() * s));
        let ref_explicit = src.poses[0].inverse().apply(&pw);
        println!("fast  : {:?}", fast.0);
        println!("expect: {:?}", ref_explicit.0);
        let diff = (fast.x() - ref_explicit.x()).abs() + (fast.y() - ref_explicit.y()).abs() + (fast.z() - ref_explicit.z()).abs();
        println!("transform diff: {diff:.6}");
    }

    // ---- 2) full sweep + metrics vs GT ----
    let mut scratch = DepthScratch::new();
    let (mut depth, conf) = geometry::sweep_depth(&images[idx].rgb, &pose, &src, &cam, &mut scratch);

    let report = |tag: &str, d: &[f32]| {
        let mut absrel = 0.0;
        let mut n = 0;
        let mut far_absrel = 0.0;
        let mut far_n = 0;
        let mut skyish = 0;
        for p in 0..npix {
            if gt.sem[p] == foundation4d::scene::SEM_SKY || gt.depth[p] >= 400.0 { continue; }
            let e = (d[p] - gt.depth[p]).abs() / gt.depth[p].max(0.5);
            absrel += e;
            n += 1;
            if gt.depth[p] > 30.0 { far_absrel += e; far_n += 1; }
            if d[p] > 130.0 { skyish += 1; }
        }
        println!("{tag}: absrel {:.4} (n={}), far {:.4} (n={}), pred>130m: {}",
            absrel / n.max(1) as f32, n, far_absrel / far_n.max(1) as f32, far_n, skyish);
    };
    report("pre-holefill", &depth);
    geometry::hole_fill(&mut depth, &conf, &pose, &cam);
    report("post-holefill", &depth);

    // Confidence distribution.
    let mut conf_hist = [0usize; 5];
    for p in 0..npix {
        let b = (conf[p] * 5.0) as usize;
        conf_hist[b.min(4)] += 1;
    }
    println!("conf hist (0-.2-.4-.6-.8-1): {:?}", conf_hist);
    // GT depth distribution.
    let mut gt_hist = [0usize; 8];
    for p in 0..npix {
        if gt.sem[p] == foundation4d::scene::SEM_SKY { continue; }
        let b = (gt.depth[p] / 20.0) as usize;
        gt_hist[b.min(7)] += 1;
    }
    println!("GT depth hist (20m bins): {:?}", gt_hist);
    // Error buckets.
    let mut eb = [0usize; 6]; // <5%, 5-15%, 15-50%, 50-200%, 200-1000%, >1000%
    for p in 0..npix {
        if gt.sem[p] == foundation4d::scene::SEM_SKY || gt.depth[p] >= 400.0 { continue; }
        let e = (depth[p] - gt.depth[p]).abs() / gt.depth[p].max(0.5);
        let b = if e < 0.05 { 0 } else if e < 0.15 { 1 } else if e < 0.5 { 2 } else if e < 2.0 { 3 } else if e < 10.0 { 4 } else { 5 };
        eb[b] += 1;
    }
    println!("error buckets <5%%/5-15/15-50/50-200/200-1000/>1000: {:?}", eb);
    // Row dump: middle row, every 8th pixel: gt vs pred.
    let row = IMG_H / 2;
    let mut line = String::new();
    for x in (0..IMG_W).step_by(16) {
        let p = row * IMG_W + x;
        if gt.sem[p] == foundation4d::scene::SEM_SKY { continue; }
        line.push_str(&format!("({}: gt {:.1} pred {:.1}) ", x, gt.depth[p], depth[p]));
    }
    println!("row {row}: {line}");
    // Vertical dump through the center column.
    let col = IMG_W / 2;
    let mut line2 = String::new();
    for y in (0..IMG_H).step_by(12) {
        let p = y * IMG_W + col;
        if gt.sem[p] == foundation4d::scene::SEM_SKY { continue; }
        line2.push_str(&format!("({}: gt {:.1} pred {:.1}) ", y, gt.depth[p], depth[p]));
    }
    println!("col {col}: {line2}");
    // Static-vs-actor split.
    {
        let mut sa = [0.0f64; 2];
        let mut sn = [0usize; 2];
        for p in 0..npix {
            if gt.sem[p] == foundation4d::scene::SEM_SKY || gt.depth[p] >= 400.0 { continue; }
            let is_actor = gt.sem[p] == foundation4d::scene::SEM_VEHICLE || gt.sem[p] == foundation4d::scene::SEM_PERSON;
            let e = (depth[p] - gt.depth[p]).abs() / gt.depth[p].max(0.5);
            sa[is_actor as usize] += e as f64;
            sn[is_actor as usize] += 1;
        }
        println!("static absrel {:.4} (n={}) | actor absrel {:.4} (n={})",
            sa[0] / sn[0].max(1) as f64, sn[0], sa[1] / sn[1].max(1) as f64, sn[1]);
    }
    // ASCII error map: per 8x8 block, worst-bucket char.
    {
        let chars = ['.', '1', '2', '3', '4', '5'];
        let mut map = String::new();
        for by in 0..IMG_H / 8 {
            for bx in 0..IMG_W / 8 {
                let mut worst = 0usize;
                let mut any = false;
                for y in 0..8 {
                    for x in 0..8 {
                        let p = (by * 8 + y) * IMG_W + bx * 8 + x;
                        if gt.sem[p] == foundation4d::scene::SEM_SKY || gt.depth[p] >= 400.0 { continue; }
                        any = true;
                        let e = (depth[p] - gt.depth[p]).abs() / gt.depth[p].max(0.5);
                        let b = if e < 0.05 { 0 } else if e < 0.15 { 1 } else if e < 0.5 { 2 } else if e < 2.0 { 3 } else if e < 10.0 { 4 } else { 5 };
                        worst = worst.max(b);
                    }
                }
                map.push(if any { chars[worst] } else { ' ' });
            }
            map.push('\n');
        }
        println!("error map (.=ok 12345=bad, 8px/block):\n{map}");
    }
    // Per-source sample dump for a bad pixel at GT depth and a far bin.
    for (yy, xx, tag) in [(72usize, 112usize, "bad"), (72usize, 144usize, "good"), (72usize, 80usize, "static"), (72usize, 160usize, "static2")] {
        let p = yy * IMG_W + xx;
        println!("--- pixel ({yy},{xx}) {tag}: sem {} rgb {:?} gt {:.1}",
            gt.sem[p], &gt.rgb[p * 3..p * 3 + 3], gt.depth[p]);
        for &d_test in &[gt.depth[p], 43.0f32, 20.0f32] {
            println!("  hypothesis d={d_test:.1}:");
            for (si, sp) in src.poses.iter().enumerate() {
                let sfr = Se3::between(sp, &pose);
                let ray = &cam.rays[p];
                let s = d_test / ray.z();
                let v3 = Vec3::new(
                    sfr.r.0[0] * ray.x() * s + sfr.r.0[1] * ray.y() * s + sfr.r.0[2] * ray.z() * s + sfr.t.x(),
                    sfr.r.0[3] * ray.x() * s + sfr.r.0[4] * ray.y() * s + sfr.r.0[5] * ray.z() * s + sfr.t.y(),
                    sfr.r.0[6] * ray.x() * s + sfr.r.0[7] * ray.y() * s + sfr.r.0[8] * ray.z() * s + sfr.t.z(),
                );
                if v3.z() < 0.1 { println!("    src{si}: behind"); continue; }
                let u = cam.fx * v3.x() / v3.z() + cam.cx - 0.5;
                let vv = cam.fy * v3.y() / v3.z() + cam.cy - 0.5;
                let col = match foundation4d::geometry::sample_rgb_pub(src.imgs[si], u, vv) {
                    Some(c) => format!("{:?}", c),
                    None => "outside".to_string(),
                };
                println!("    src{si}: proj ({u:.1},{vv:.1}) col {col}");
            }
        }
    }
    // Cost-curve dump for selected pixels: raw cost, SGM-aggregated, GT bin.
    for (yy, xx) in [(72usize, 112usize), (72usize, 144usize), (72usize, 80usize), (60usize, 96usize)] {
        let p = yy * IMG_W + xx;
        let qc = (yy / 2) * (IMG_W / 2) + xx / 2;
        let gt_d = gt.depth[p];
        let k_gt = (gt_d / D_MIN).ln() / ((D_MAX / D_MIN).ln() / 31.0);
        let (cost_row, sgm_row, d_half, c_half) = geometry::debug_coarse_row(&scratch, qc);
        println!("pixel ({yy},{xx}) gt {:.1} (bin {:.1}) d_half {:.1} conf_half {:.3}:", gt_d, k_gt, d_half, c_half);
        let mut line = String::new();
        for k in 0..32 {
            line.push_str(&format!("{}:{:.0}/{:.0} ", k, cost_row[k], sgm_row[k]));
            if (k + 1) % 8 == 0 { println!("  bins {:?}: {line}", k - 7..=k); line.clear(); }
        }
    }
    let _ = (D_MIN, D_MAX);
}
