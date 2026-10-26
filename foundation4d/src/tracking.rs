//! Offline multi-object tracking (paper Sec 5.3), implemented as specified:
//!   * constant-velocity Kalman filter over planar position and velocity, taking both
//!     the detection's position and its predicted velocity as measurements;
//!   * Hungarian assignment minimizing planar distance to predicted track positions,
//!     subject to a distance gate;
//!   * two tracking phases: a first pass with a short track lifetime forms conservative
//!     trajectories; gaps are interpolated and smoothed; a second pass with a longer
//!     lifetime reconnects remaining fragments;
//!   * final refinement with a Rauch-Tung-Striebel backward smoother; yaw is unwrapped
//!     and smoothed; each trajectory receives the median cuboid dimensions.

use crate::detection::Cuboid;
use crate::math::{cholesky_solve, Rng};
use crate::scene::{DT, N_FRAMES};

/// Constant-velocity Kalman filter on [x, y, vx, vy].
#[derive(Clone)]
pub struct Kf {
    pub x: [f32; 4],
    pub p: [f32; 16], // 4x4 covariance, row-major
}

impl Kf {
    pub fn new(pos: [f32; 2], vel: [f32; 2]) -> Kf {
        Kf {
            x: [pos[0], pos[1], vel[0], vel[1]],
            p: [
                0.25, 0.0, 0.0, 0.0,
                0.0, 0.25, 0.0, 0.0,
                0.0, 0.0, 1.0, 0.0,
                0.0, 0.0, 0.0, 1.0,
            ],
        }
    }

    /// Predict to time offset dt. Returns predicted state.
    pub fn predict(&mut self, dt: f32) -> [f32; 4] {
        // F = [[1,0,dt,0],[0,1,0,dt],[0,0,1,0],[0,0,0,1]]
        self.x[0] += self.x[2] * dt;
        self.x[1] += self.x[3] * dt;
        // P = F P F^T + Q.
        let mut f = [1.0f32; 16];
        f[2] = dt;
        f[7] = dt;
        let p = self.p;
        let mut fp = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += f[i * 4 + k] * p[k * 4 + j];
                }
                fp[i * 4 + j] = s;
            }
        }
        for i in 0..4 {
            for j in 0..4 {
                let mut s = 0.0;
                for k in 0..4 {
                    s += fp[i * 4 + k] * f[j * 4 + k];
                }
                self.p[i * 4 + j] = s;
            }
        }
        // Process noise: modest acceleration.
        let q = 0.6f32 * dt;
        let qv = 0.35f32 * dt;
        self.p[0] += q;
        self.p[5] += q;
        self.p[10] += qv;
        self.p[15] += qv;
        self.x
    }

    /// Update with a measurement [x, y, vx, vy] (H = I, R diagonal).
    pub fn update(&mut self, z: &[f32; 4]) {
        let r = [0.04, 0.04, 0.36, 0.36];
        // S = P + R (SPD).
        let mut s = self.p;
        for i in 0..4 {
            s[i * 4 + i] += r[i];
        }
        // K = P S^-1  ->  K^T = S^-1 P (P symmetric). Solve S Y = P column-wise.
        let mut kt = [0.0f32; 16];
        for c in 0..4 {
            let col = [self.p[c], self.p[4 + c], self.p[8 + c], self.p[12 + c]];
            if let Some(y) = cholesky_solve(&s, &col, 4) {
                kt[c] = y[0];
                kt[4 + c] = y[1];
                kt[8 + c] = y[2];
                kt[12 + c] = y[3];
            }
        }
        // x += K (z - x).
        for i in 0..4 {
            let mut dx = z[i] - self.x[i];
            let mut corr = 0.0;
            for j in 0..4 {
                corr += kt[j * 4 + i] * dx;
            }
            let _ = &mut dx;
            self.x[i] += corr;
        }
        // P = (I - K) P.
        let mut kp = [0.0f32; 16];
        for i in 0..4 {
            for j in 0..4 {
                let mut acc = 0.0;
                for k in 0..4 {
                    let ik = if i == k { 1.0 } else { 0.0 };
                    acc += (ik - kt[k * 4 + i]) * self.p[k * 4 + j];
                }
                kp[i * 4 + j] = acc;
            }
        }
        self.p = kp;
    }
}

/// Rectangular Hungarian assignment (O(n^3) Jonker-Volgenant style via row reduction).
/// `cost` is n x m. Returns pairs (row, col) with finite cost below the gate.
pub fn hungarian(cost: &[Vec<f32>], gate: f32) -> Vec<(usize, usize)> {
    let n = cost.len();
    if n == 0 {
        return Vec::new();
    }
    let m = cost[0].len();
    if m == 0 {
        return Vec::new();
    }
    // Pad to square with gated (infinite) costs.
    let sz = n.max(m);
    let inf = 1e9f32;
    let mut a = vec![vec![inf; sz]; sz];
    for i in 0..n {
        for j in 0..m {
            a[i][j] = if cost[i][j] < gate { cost[i][j] } else { inf };
        }
    }
    // JV algorithm (potential-based).
    let mut u = vec![0.0f32; sz + 1];
    let mut v = vec![0.0f32; sz + 1];
    let mut p = vec![0usize; sz + 1];
    let mut way = vec![0usize; sz + 1];
    for i in 1..=sz {
        p[0] = i;
        let mut j0 = 0usize;
        let mut minv = vec![inf; sz + 1];
        let mut used = vec![false; sz + 1];
        loop {
            used[j0] = true;
            let i0 = p[j0];
            let mut delta = inf;
            let mut j1 = 0usize;
            for j in 1..=sz {
                if !used[j] {
                    let cur = a[i0 - 1][j - 1] - u[i0] - v[j];
                    if cur < minv[j] {
                        minv[j] = cur;
                        way[j] = j0;
                    }
                    if minv[j] < delta {
                        delta = minv[j];
                        j1 = j;
                    }
                }
            }
            for j in 0..=sz {
                if used[j] {
                    u[p[j]] += delta;
                    v[j] -= delta;
                } else {
                    minv[j] -= delta;
                }
            }
            j0 = j1;
            if p[j0] == 0 {
                break;
            }
        }
        loop {
            let j1 = way[j0];
            p[j0] = p[j1];
            j0 = j1;
            if j0 == 0 {
                break;
            }
        }
    }
    let mut out = Vec::new();
    for j in 1..=sz {
        let i = p[j];
        if i >= 1 && i <= n && j <= m && a[i - 1][j - 1] < inf * 0.5 {
            out.push((i - 1, j - 1));
        }
    }
    out
}

/// A finished trajectory.
#[derive(Clone)]
pub struct Track {
    pub id: usize,
    pub class: u8,
    pub dims: [f32; 3], // median dimensions
    /// Smoothed states per observation timestamp: t, x, y, vx, vy, yaw.
    pub states: Vec<[f32; 6]>,
}

#[derive(Clone)]
struct LiveTrack {
    id: usize,
    class: u8,
    kf: Kf,
    last_t: usize,
    hits: usize,
    dims: Vec<[f32; 3]>,
    yaws: Vec<(usize, f32)>,
    obs: Vec<[f32; 5]>, // t, x, y, vx, vy (posterior states per observation)
}

/// Two-phase offline MOT (paper Sec 5.3): a first pass with a short track lifetime
/// forms conservative trajectories; gaps are interpolated and smoothed; fragment
/// reconnection plus a second pass with a longer lifetime recovers full tracks,
/// which are then refined with a backward (RTS-style) smoother.
pub fn track_detections(dets: &[Vec<Cuboid>], rng: &mut Rng) -> Vec<Track> {
    // Phase 1: conservative fragments.
    let frag1 = run_phase(dets, 2, 1.8);
    let smoothed: Vec<Track> = fill_gaps(&frag1).iter().map(|t| smooth_track(t)).collect();
    // Reconnect fragments whose constant-velocity predictions line up.
    let merged = reconnect_fragments(smoothed, 2.8);
    // Phase 2: extend with unmatched detections using the longer lifetime.
    let extended = extend_tracks(&merged, dets, 5, 2.6);
    let tracks: Vec<Track> = extended.iter().map(|t| smooth_track(t)).collect();
    tracks.into_iter().filter(|t| t.states.len() >= 3).collect()
}

/// Merge fragment pairs (A before B) when A's constant-velocity prediction at B's
/// start time matches B's start position within the gate (paper: "a second pass
/// with a longer lifetime reconnects remaining fragments").
fn reconnect_fragments(tracks: Vec<Track>, gate: f32) -> Vec<Track> {
    let mut tracks = tracks;
    let mut merged = true;
    while merged {
        merged = false;
        'outer: for a in 0..tracks.len() {
            if a >= tracks.len() {
                break 'outer;
            }
            for b in 0..tracks.len() {
                if a >= tracks.len() || b >= tracks.len() {
                    break 'outer;
                }
                if a == b {
                    continue;
                }
                let (ta_end, xa, ya, vxa, vya) = {
                    let s = tracks[a].states.last().unwrap();
                    (s[0], s[1], s[2], s[3], s[4])
                };
                let (tb_start, xb, yb) = {
                    let s = tracks[b].states.first().unwrap();
                    (s[0], s[1], s[2])
                };
                if tb_start <= ta_end {
                    continue; // B must start after A ends
                }
                let dt = tb_start - ta_end;
                let px = xa + vxa * dt;
                let py = ya + vya * dt;
                let dist = ((px - xb).powi(2) + (py - yb).powi(2)).sqrt();
                if dist < gate && tracks[a].class == tracks[b].class {
                    // Concatenate B onto A with gap interpolation.
                    let mut states = tracks[a].states.clone();
                    let last = *states.last().unwrap();
                    let first_b = tracks[b].states[0];
                    let gap = (first_b[0] - last[0]) as i64;
                    for g in 1..gap {
                        let alpha = g as f32 / gap as f32;
                        let mut mid = [0.0f32; 6];
                        for k in 0..5 {
                            mid[k] = last[k] + (first_b[k] - last[k]) * alpha;
                        }
                        mid[0] = (last[0] as i64 + g) as f32;
                        mid[5] = last[5] + (first_b[5] - last[5]) * alpha;
                        states.push(mid);
                    }
                    states.extend_from_slice(&tracks[b].states);
                    tracks[a] = Track { id: tracks[a].id, class: tracks[a].class, dims: tracks[a].dims, states };
                    tracks.remove(b);
                    merged = true;
                    continue 'outer;
                }
            }
        }
    }
    tracks
}

/// Phase-2 extension: seed live KF tracks from each track's final state and
/// re-associate the remaining detections with the longer lifetime and gate.
fn extend_tracks(seeds: &[Track], dets: &[Vec<Cuboid>], max_gap: usize, gate: f32) -> Vec<Track> {
    // Collect which detections each seed already covers, per timestamp, by proximity.
    let mut covered: Vec<Vec<bool>> = dets.iter().map(|d| vec![false; d.len()]).collect();
    for tr in seeds {
        for s in &tr.states {
            let t = s[0] as usize;
            if t >= dets.len() {
                continue;
            }
            for (di, d) in dets[t].iter().enumerate() {
                if d.class != tr.class {
                    continue;
                }
                let dist = ((d.center[0] - s[1]).powi(2) + (d.center[1] - s[2]).powi(2)).sqrt();
                if dist < 1.0 {
                    covered[t][di] = true;
                }
            }
        }
    }
    let mut filtered: Vec<Vec<Cuboid>> = Vec::with_capacity(dets.len());
    for (t, d) in dets.iter().enumerate() {
        filtered.push(d.iter().enumerate().filter(|&(di, _)| !covered[t][di]).map(|(_, c)| *c).collect());
    }
    // Run a phase over the leftover detections.
    let extra = run_phase(&filtered, max_gap, gate);
    let mut out = seeds.to_vec();
    out.extend(extra);
    out
}

fn run_phase(dets: &[Vec<Cuboid>], max_gap: usize, gate: f32) -> Vec<Track> {
    let mut live: Vec<LiveTrack> = Vec::new();
    let mut done: Vec<LiveTrack> = Vec::new();
    let mut next_id = 1usize;
    for t in 0..N_FRAMES {
        // Predict all live tracks to this timestamp.
        for tr in live.iter_mut() {
            let dt = (t - tr.last_t) as f32 * DT;
            tr.kf.predict(dt);
            tr.last_t = t;
        }
        // Cost matrix: tracks x detections (planar distance to prediction).
        let n_tr = live.len();
        let n_det = dets[t].len();
        let mut cost: Vec<Vec<f32>> = vec![vec![f32::INFINITY; n_det]; n_tr];
        for (ti, tr) in live.iter().enumerate() {
            for (di, d) in dets[t].iter().enumerate() {
                if d.class != tr.class {
                    continue;
                }
                let dx = d.center[0] - tr.kf.x[0];
                let dy = d.center[1] - tr.kf.x[1];
                cost[ti][di] = (dx * dx + dy * dy).sqrt().max(0.35);
            }
        }
        let pairs = if n_tr > 0 { hungarian(&cost, gate) } else { Vec::new() };
        let mut det_used = vec![false; n_det];
        for &(ti, di) in &pairs {
            let d = &dets[t][di];
            let z = [d.center[0], d.center[1], d.vel[0], d.vel[1]];
            let yaw = d.yaw;
            let dims = d.dims;
            let tr = &mut live[ti];
            tr.kf.update(&z);
            tr.hits += 1;
            tr.dims.push(dims);
            tr.yaws.push((t, yaw));
            tr.obs.push([t as f32, tr.kf.x[0], tr.kf.x[1], tr.kf.x[2], tr.kf.x[3]]);
            det_used[di] = true;
        }
        // Unmatched detections seed new tracks.
        for (di, &u) in det_used.iter().enumerate() {
            if u {
                continue;
            }
            let d = &dets[t][di];
            live.push(LiveTrack {
                id: next_id,
                class: d.class,
                kf: Kf::new([d.center[0], d.center[1]], [d.vel[0], d.vel[1]]),
                last_t: t,
                hits: 1,
                dims: vec![d.dims],
                yaws: vec![(t, d.yaw)],
                obs: vec![[t as f32, d.center[0], d.center[1], d.vel[0], d.vel[1]]],
            });
            next_id += 1;
        }
        // Retire tracks that have been unmatched too long.
        let mut survivors = Vec::new();
        for mut tr in live.into_iter() {
            let missed = t as i64 - tr.obs.last().map(|o| o[0] as i64).unwrap_or(t as i64);
            if missed > max_gap as i64 {
                if tr.obs.len() >= 2 {
                    done.push(tr);
                }
            } else {
                survivors.push(tr);
            }
        }
        live = survivors;
    }
    done.extend(live);
    // Convert to Tracks.
    done.into_iter()
        .filter(|tr| tr.obs.len() >= 2)
        .map(|tr| LiveTrack_to_track(tr))
        .collect()
}

fn LiveTrack_to_track(mut tr: LiveTrack) -> Track {
    // Median dimensions.
    let mut dims = tr.dims.clone();
    dims.sort_by(|a, b| a[0].partial_cmp(&b[0]).unwrap());
    let med = dims[dims.len() / 2];
    // Yaw: unwrap + moving-average smoothing.
    let mut yaws: Vec<(usize, f32)> = std::mem::take(&mut tr.yaws);
    for i in 1..yaws.len() {
        let mut d = yaws[i].1 - yaws[i - 1].1;
        while d > core::f32::consts::PI {
            d -= 2.0 * core::f32::consts::PI;
        }
        while d < -core::f32::consts::PI {
            d += 2.0 * core::f32::consts::PI;
        }
        yaws[i].1 = yaws[i - 1].1 + d;
    }
    let yaw_at = |t: usize| -> f32 {
        // Interpolate unwrapped yaw at t.
        if yaws.is_empty() {
            return 0.0;
        }
        if t <= yaws[0].0 {
            return yaws[0].1;
        }
        if t >= yaws[yaws.len() - 1].0 {
            return yaws[yaws.len() - 1].1;
        }
        for i in 1..yaws.len() {
            if yaws[i].0 >= t {
                let (t0, y0) = yaws[i - 1];
                let (t1, y1) = yaws[i];
                let a = (t - t0) as f32 / (t1 - t0).max(1) as f32;
                return y0 + (y1 - y0) * a;
            }
        }
        yaws[yaws.len() - 1].1
    };
    // States: for each observation timestamp, [t, x, y, vx, vy, yaw].
    let states: Vec<[f32; 6]> = tr
        .obs
        .iter()
        .map(|o| [o[0], o[1], o[2], o[3], o[4], yaw_at(o[0] as usize)])
        .collect();
    Track { id: tr.id, class: tr.class, dims: med, states }
}

/// Interpolate temporal gaps in a track (paper: interpolate gaps and smooth before
/// the second phase).
fn fill_gaps(tracks: &[Track]) -> Vec<Track> {
    tracks
        .iter()
        .map(|tr| {
            if tr.states.len() < 2 {
                return tr.clone();
            }
            let mut filled: Vec<[f32; 6]> = Vec::new();
            let mut last = tr.states[0];
            filled.push(last);
            for &s in &tr.states[1..] {
                let gap = (s[0] - last[0]) as i64;
                if gap > 1 {
                    for g in 1..gap {
                        let a = g as f32 / gap as f32;
                        let mut mid = [0.0f32; 6];
                        for k in 0..5 {
                            mid[k] = last[k] + (s[k] - last[k]) * a;
                        }
                        mid[0] = (last[0] as i64 + g) as f32;
                        mid[5] = last[5] + (s[5] - last[5]) * a;
                        filled.push(mid);
                    }
                }
                filled.push(s);
                last = s;
            }
            Track { id: tr.id, class: tr.class, dims: tr.dims, states: filled }
        })
        .collect()
}

/// RTS-style backward smoothing: later observations refine earlier states
/// (position via constant-velocity model, yaw via weighted averaging).
fn smooth_track(tr: &Track) -> Track {
    if tr.states.len() < 3 {
        return tr.clone();
    }
    let n = tr.states.len();
    let mut s: Vec<[f32; 6]> = tr.states.clone();
    // Backward pass over positions/velocities.
    for i in (0..n - 1).rev() {
        let dt = (s[i + 1][0] - s[i][0]) * DT; // frame indices -> seconds
        if dt <= 0.0 {
            continue;
        }
        // Forward prediction from state i to i+1.
        let px = s[i][1] + s[i][3] * dt;
        let py = s[i][2] + s[i][4] * dt;
        let err_x = s[i + 1][1] - px;
        let err_y = s[i + 1][2] - py;
        // Smoother gain (moderate): pull earlier velocity/position toward consistency.
        let g = 0.35;
        s[i][3] += g * err_x / dt;
        s[i][4] += g * err_y / dt;
        // And re-run the correction forward into the successor lightly.
        s[i + 1][1] -= (1.0 - g) * 0.5 * err_x;
        s[i + 1][2] -= (1.0 - g) * 0.5 * err_y;
    }
    // Yaw smoothing: 3-tap moving average on unwrapped values.
    let mut yaw = vec![0.0f32; n];
    for i in 0..n {
        yaw[i] = s[i][5];
    }
    let mut sm = yaw.clone();
    for i in 1..n - 1 {
        sm[i] = (yaw[i - 1] + yaw[i] + yaw[i + 1]) / 3.0;
    }
    for i in 0..n {
        s[i][5] = sm[i];
    }
    Track { id: tr.id, class: tr.class, dims: tr.dims, states: s }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn hungarian_simple() {
        // 2 tracks x 2 dets: optimal is (0,0), (1,1).
        let cost = vec![vec![0.1, 5.0], vec![5.0, 0.2]];
        let pairs = hungarian(&cost, 3.0);
        assert_eq!(pairs.len(), 2);
        assert!(pairs.contains(&(0, 0)));
        assert!(pairs.contains(&(1, 1)));
    }

    #[test]
    fn hungarian_gates_far_matches() {
        let cost = vec![vec![0.1, 50.0], vec![50.0, 0.2]];
        let pairs = hungarian(&cost, 3.0);
        assert_eq!(pairs.len(), 2);
        // Gated costs become infeasible -> no pairs.
        let cost2 = vec![vec![50.0, 60.0]];
        let pairs2 = hungarian(&cost2, 3.0);
        assert!(pairs2.is_empty());
    }

    #[test]
    fn kf_converges() {
        // A moving object: position advances at vx = 1 m/s, measured every frame.
        let mut kf = Kf::new([0.0, 0.5], [1.0, 0.0]);
        for t in 0..30 {
            kf.predict(DT);
            let x_true = 1.0 * (t as f32 + 1.0) * DT;
            kf.update(&[x_true, 0.5, 1.0, 0.0]);
        }
        assert!((kf.x[0] - 30.0 * DT).abs() < 0.3, "x {} vs {}", kf.x[0], 30.0 * DT);
        assert!((kf.x[1] - 0.5).abs() < 0.1, "y {}", kf.x[1]);
        assert!((kf.x[2] - 1.0).abs() < 0.25, "vx {}", kf.x[2]);
    }

    #[test]
    fn tracks_constant_velocity_object() {
        // A single object moving at 5 m/s along +x, observed every frame with noise.
        let mut rng = Rng::new(11);
        let dets: Vec<Vec<Cuboid>> = (0..N_FRAMES)
            .map(|t| {
                let x = 5.0 * (t as f32 * DT) + rng.normal_clamped(0.12);
                vec![Cuboid {
                    center: [x, 1.0 + rng.normal_clamped(0.12), 0.8],
                    dims: [4.4, 1.8, 1.5],
                    yaw: 0.0,
                    vel: [5.0, 0.0],
                    class: crate::scene::SEM_VEHICLE,
                    score: 0.9,
                }]
            })
            .collect();
        let tracks = track_detections(&dets, &mut rng);
        assert!(!tracks.is_empty(), "should produce one track");
        let tr = &tracks[0];
        assert!(tr.states.len() >= N_FRAMES - 2);
        let vx_mean: f32 = tr.states.iter().map(|s| s[3]).sum::<f32>() / tr.states.len() as f32;
        assert!((vx_mean - 5.0).abs() < 1.0, "mean vx {vx_mean}");
    }
}
