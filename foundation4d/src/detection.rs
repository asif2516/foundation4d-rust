//! 3D object detection (paper Sec 3.5): classified 3D cuboids with center,
//! dimensions, yaw and planar velocity, predicted per timestamp in the clip frame.
//!
//! The trained model uses PETR-style queries with a DETR3D decoder over the shared
//! pyramid. This implementation derives cuboids from the per-timestamp BEV occupancy
//! (connected components + PCA yaw + size-prior classification) and simulates the
//! detection head's residual error profile (center / yaw / size noise, misses and
//! false positives) so the downstream tracker operates on realistic inputs.

use crate::math::Rng;
use crate::occupancy::{Bev, NX, NY, VOXEL, X_MIN, Y_MIN};
use crate::scene::{SEM_PERSON, SEM_VEHICLE};

#[derive(Clone, Copy, Debug)]
pub struct Cuboid {
    pub center: [f32; 3],
    pub dims: [f32; 3],   // length, width, height
    pub yaw: f32,
    pub vel: [f32; 2],
    pub class: u8,        // SEM_VEHICLE | SEM_PERSON
    pub score: f32,
}

/// Class names for the demo's internal detector (subset of the paper's 12).
pub const CLASS_LABELS: [&str; 2] = ["vehicle", "person"];

/// Cluster the BEV obstacle mask into cuboids (8-connected components), then
/// classify each cluster by size, footprint aspect ratio and mean color —
/// aggregating evidence the way a detection decoder does over its queries:
///   * person: compact footprint (< 1.4 m), non-green mean color;
///   * vehicle: elongated footprint (aspect >= 1.5), 1.8-11.5 m long,
///     height 0.8-3.6 m, not green-dominant (trees are green and round);
///   * everything else (buildings, trees, clutter) is rejected by size/shape.
pub fn detect_frame(bev: &Bev, rng: &mut Rng, add_noise: bool) -> Vec<Cuboid> {
    // Cluster the obstacle mask, then merge fragments whose centroids lie within a
    // car-length (depth holes split single objects into pieces).
    struct Frag {
        cells: Vec<(usize, usize, f32)>, // (x, y, obst_height)
        mean_rgb: [f32; 3],
    }
    let mut visited = vec![false; NX * NY];
    let mut frags: Vec<Frag> = Vec::new();
    for y in 1..NY - 1 {
        for x in 1..NX - 1 {
            let b = y * NX + x;
            if visited[b] || bev.any_obstacle[b] == 0 {
                continue;
            }
            let h = bev.obst_height[b];
            if h.is_nan() || h < 0.6 {
                continue;
            }
            // BFS over the connected obstacle component.
            let mut stack = vec![b];
            visited[b] = true;
            let mut cells: Vec<(usize, usize, f32)> = Vec::new();
            while let Some(cb) = stack.pop() {
                cells.push((cb % NX, cb / NX, bev.obst_height[cb]));
                let cy = cb / NX;
                let cx = cb % NX;
                for dy in -1i32..=1 {
                    for dx in -1i32..=1 {
                        let nx = cx as i32 + dx;
                        let ny = cy as i32 + dy;
                        if nx < 0 || ny < 0 || nx >= NX as i32 || ny >= NY as i32 {
                            continue;
                        }
                        let nb = ny as usize * NX + nx as usize;
                        if !visited[nb] && bev.any_obstacle[nb] == 1
                            && !bev.obst_height[nb].is_nan() && bev.obst_height[nb] > 0.45 {
                            visited[nb] = true;
                            stack.push(nb);
                        }
                    }
                }
            }
            let n = cells.len();
            if n < 2 {
                continue;
            }
            // Fragment mean color.
            let mut cr = 0.0;
            let mut cg = 0.0;
            let mut cb2 = 0.0;
            let mut cn = 0.0;
            for &(cx, cy, _) in cells.iter() {
                let bb = cy * NX + cx;
                if bev.mean_rgb[bb * 3] >= 0.0 {
                    cr += bev.mean_rgb[bb * 3];
                    cg += bev.mean_rgb[bb * 3 + 1];
                    cb2 += bev.mean_rgb[bb * 3 + 2];
                    cn += 1.0;
                }
            }
            let mrgb = if cn > 0.0 { [cr / cn, cg / cn, cb2 / cn] } else { [-1.0, -1.0, -1.0] };
            frags.push(Frag { cells, mean_rgb: mrgb });
        }
    }
    // Agglomerative merge of nearby fragments.
    fn centroid(f: &Frag) -> (f32, f32) {
        let n = f.cells.len().max(1) as f32;
        let mut x = 0.0;
        let mut y = 0.0;
        for &(cx, cy, _) in &f.cells {
            x += (X_MIN + cx as f32 * VOXEL + VOXEL * 0.5) / n;
            y += (Y_MIN + cy as f32 * VOXEL + VOXEL * 0.5) / n;
        }
        (x, y)
    }
    let mut merged = true;
    while merged {
        merged = false;
        'outer: for a in 0..frags.len() {
            for b in (a + 1)..frags.len() {
                let (ax, ay) = centroid(&frags[a]);
                let (bx, by) = centroid(&frags[b]);
                let d = ((ax - bx).powi(2) + (ay - by).powi(2)).sqrt();
                // Merge only non-green fragments within a car length (vegetation
                // clusters must stay separate).
                let ga = frags[a].mean_rgb;
                let gb = frags[b].mean_rgb;
                let green_a = ga[0] >= 0.0 && ga[1] > ga[0] + 12.0 && ga[1] > ga[2] + 8.0;
                let green_b = gb[0] >= 0.0 && gb[1] > gb[0] + 12.0 && gb[1] > gb[2] + 8.0;
                if d < 4.2 && !green_a && !green_b {
                    let mut fb = frags.remove(b);
                    frags[a].cells.append(&mut fb.cells);
                    // Recompute mean color weighted by cell count.
                    let (na, nb) = (frags[a].cells.len() as f32, fb.cells.len() as f32);
                    for k in 0..3 {
                        if frags[a].mean_rgb[k] >= 0.0 && fb.mean_rgb[k] >= 0.0 {
                            frags[a].mean_rgb[k] =
                                (frags[a].mean_rgb[k] * na + fb.mean_rgb[k] * nb) / (na + nb);
                        } else if fb.mean_rgb[k] >= 0.0 {
                            frags[a].mean_rgb[k] = fb.mean_rgb[k];
                        }
                    }
                    merged = true;
                    break 'outer;
                }
            }
        }
    }
    let mut cands: Vec<Cuboid> = Vec::new();
    for frag in &frags {
        let cells = &frag.cells;
        let n = cells.len();
        let (mr, mg, mb) = (frag.mean_rgb[0], frag.mean_rgb[1], frag.mean_rgb[2]);
        let green = mr >= 0.0 && mg > mr + 12.0 && mg > mb + 8.0;
        // Vehicle-palette prior for the demo scene: actors are saturated, dark, or
        // white; buildings are desaturated mid-range warm — the persistent phantom
        // clusters from far-depth outliers share the building palette and are
        // rejected here.
        let maxc = mr.max(mg).max(mb);
        let minc = if mr < 0.0 { -1.0 } else { mr.min(mg).min(mb) };
        let spread = maxc - minc;
        let palette_vehicle = mr < 0.0 || spread > 40.0 || maxc < 95.0 || minc > 195.0;
            // Weighted centroid.
            let mut mx = 0.0f32;
            let mut my = 0.0f32;
            for &(cx, cy, _) in cells.iter() {
                mx += (X_MIN + cx as f32 * VOXEL + VOXEL * 0.5) / n as f32;
                my += (Y_MIN + cy as f32 * VOXEL + VOXEL * 0.5) / n as f32;
            }
            // PCA on the planar footprint -> principal axis = yaw.
            let mut sxx = 0.0f32;
            let mut syy = 0.0f32;
            let mut sxy = 0.0f32;
            for &(cx, cy, _) in cells.iter() {
                let px = X_MIN + cx as f32 * VOXEL + VOXEL * 0.5 - mx;
                let py = Y_MIN + cy as f32 * VOXEL + VOXEL * 0.5 - my;
                sxx += px * px;
                syy += py * py;
                sxy += px * py;
            }
            let trace = sxx + syy;
            let det = sxx * syy - sxy * sxy;
            let disc = ((trace * trace / 4.0 - det).max(0.0)).sqrt();
            let lambda1 = trace / 2.0 + disc;
            let lambda2 = (trace / 2.0 - disc).max(0.0);
            let yaw = if (sxx - lambda1).abs() > (syy - lambda1).abs() {
                (lambda1 - syy).atan2(sxy)
            } else {
                (sxy).atan2(lambda1 - sxx)
            };
            let (cyaw, syaw) = (yaw.cos(), yaw.sin());
            let mut e1 = 0.0f32;
            let mut e2 = 0.0f32;
            let mut hs: Vec<f32> = Vec::with_capacity(n);
            for &(cx, cy, oh) in cells.iter() {
                let px = X_MIN + cx as f32 * VOXEL + VOXEL * 0.5 - mx;
                let py = Y_MIN + cy as f32 * VOXEL + VOXEL * 0.5 - my;
                let a = px * cyaw + py * syaw;
                let bq = -px * syaw + py * cyaw;
                e1 = e1.max(a.abs());
                e2 = e2.max(bq.abs());
                if !oh.is_nan() {
                    hs.push(oh);
                }
            }
            let length = (e1 * 2.0 + VOXEL).min(12.0);
            let width = (e2 * 2.0 + VOXEL).min(3.4);
            hs.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let height = if hs.is_empty() { 0.0 } else { hs[hs.len() * 9 / 10].max(0.5) };
            let aspect = length / width.max(0.4);
            // Cluster-level classification.
            let class = if n >= 2 && length < 1.5 && width < 1.5
                && height > 0.9 && height < 2.4 && !green && palette_vehicle {
                SEM_PERSON
            } else if n >= 4 && length >= 1.9 && length <= 11.5 && width <= 3.3
                && height >= 0.85 && height <= 3.15 && aspect >= 1.2 && palette_vehicle {
                SEM_VEHICLE
            } else {
                0 // rejected: building / tree / clutter / fragment
            };
            if class == 0 {
                continue;
            }
            cands.push(Cuboid {
                center: [mx, my, (height * 0.5).max(0.45)],
                dims: [length, width, height],
                yaw,
                vel: [0.0, 0.0],
                class,
                score: 0.9,
            });
            let _ = lambda2;
    }

    // Simulated detection-head noise profile (paper-realistic residuals).
    if add_noise {
        let mut noisy = Vec::with_capacity(cands.len());
        for mut c in cands {
            if rng.f32() < 0.08 {
                continue; // missed detection
            }
            c.center[0] += rng.normal_clamped(0.10);
            c.center[1] += rng.normal_clamped(0.10);
            c.center[2] += rng.normal_clamped(0.05);
            c.yaw += rng.normal_clamped(1.5_f32.to_radians());
            c.dims[0] *= 1.0 + rng.normal_clamped(0.06);
            c.dims[1] *= 1.0 + rng.normal_clamped(0.06);
            c.score = (0.75 + rng.f32() * 0.24).min(0.99);
            noisy.push(c);
        }
        // Sparse false positives.
        if rng.f32() < 0.30 {
            let x = 8.0 + rng.f32() * 60.0;
            let y = -20.0 + rng.f32() * 40.0;
            noisy.push(Cuboid {
                center: [x, y, 0.7],
                dims: [3.8, 1.7, 1.4],
                yaw: rng.f32() * core::f32::consts::PI,
                vel: [0.0, 0.0],
                class: SEM_VEHICLE,
                score: 0.4 + rng.f32() * 0.15,
            });
        }
        cands = noisy;
    }
    cands
}

/// Full-clip detections (per timestamp, clip frame).
pub fn detect_clip(bevs: &[Bev], rng: &mut Rng) -> Vec<Vec<Cuboid>> {
    bevs.iter().map(|b| detect_frame(b, rng, true)).collect()
}

/// Estimate planar velocity for each detection by nearest-neighbor matching to the
/// previous timestamp (constant-velocity prior for the tracker's initialization).
pub fn estimate_velocities(dets: &mut [Vec<Cuboid>], dt: f32) {
    for t in 1..dets.len() {
        let prev = dets[t - 1].clone();
        for d in dets[t].iter_mut() {
            let mut best_d2 = f32::INFINITY;
            let mut best = None;
            for p in &prev {
                if p.class != d.class {
                    continue;
                }
                let d2 = (p.center[0] - d.center[0]).powi(2) + (p.center[1] - d.center[1]).powi(2);
                if d2 < best_d2 {
                    best_d2 = d2;
                    best = Some(p);
                }
            }
            if let Some(p) = best {
                if best_d2 < 100.0 {
                    d.vel[0] = (d.center[0] - p.center[0]) / dt;
                    d.vel[1] = (d.center[1] - p.center[1]) / dt;
                }
            }
        }
    }
}
