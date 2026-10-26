//! Video inference assembly (paper Sec 5): overlapping windows, sequential window
//! stitching (Eq. 14) and global pose graph optimization (Eq. 15).
//!
//! PGO estimates one reference-camera trajectory for the whole clip from raw window
//! predictions: each unique reference-camera sample is a node shared by all windows
//! containing it; within-window relative poses are separate factors; rotations are
//! optimized first (SO(3) Gauss-Newton with rotation-vector updates), then
//! translations solve a linear least-squares problem with the paper's motion prior
//! (lambda_motion = 30). Factor weights decrease linearly from 1 at the window's
//! beginning to 0.1 at its end; factors connect samples separated by 1, 2, or 3 rows
//! (the paper uses 1, 2, 4, 8 for 8-frame windows — scaled to our 4-frame windows).

use crate::math::{Mat3, Quat, Rng, Se3, Vec3, cholesky_solve};

pub const LAMBDA_MOTION: f32 = 10.0;

/// A window's predicted ego trajectory (poses in the clip frame, noisy).
#[derive(Clone)]
pub struct WindowPred {
    pub frames: Vec<usize>, // clip-frame indices covered by this window
    pub poses: Vec<Se3>,    // predicted reference-camera pose per frame (clip frame)
}

/// One pose-graph factor: relative measurement from window k between nodes i < j.
#[derive(Clone, Copy)]
pub struct Factor {
    pub i: usize,
    pub j: usize,
    pub z: Se3,     // measurement (R_ij, t_ij) mapping coords from j to i
    pub weight: f32,
}

/// Build factors from all windows (separations 1..=3, linearly decaying weights).
pub fn build_factors(windows: &[WindowPred]) -> Vec<Factor> {
    let mut out = Vec::new();
    for w in windows {
        let n = w.frames.len();
        for a in 0..n {
            for b in (a + 1)..n {
                let sep = b - a;
                if sep > 3 {
                    continue;
                }
                let i = w.frames[a];
                let j = w.frames[b];
                // Weight: linear from 1 at the window's beginning to 0.1 at its end,
                // evaluated at the pair's midpoint (paper Sec 5.2).
                let mid = (a + b) as f32 * 0.5;
                let wgt = 1.0 - 0.9 * (mid / (n as f32 - 1.0)).clamp(0.0, 1.0);
                let z = Se3::between(&w.poses[a], &w.poses[b]);
                out.push(Factor { i, j, z, weight: wgt });
            }
        }
    }
    out
}

/// Global pose graph optimization (Eq. 15). Returns the optimized trajectory
/// (first pose fixed to identity).
///
/// Stage 1 minimizes sum w ||R_i^T R_j - R_ij||_F^2 over SO(3) by iterative weighted
/// geodesic averaging (each node relaxes toward the weighted quaternion mean of the
/// rotation targets implied by its factors; the paper refines with L-BFGS using
/// rotation-vector updates — same objective, different optimizer).
/// Stage 2 solves the translation least-squares system — rotations fixed, the normal
/// equations reduce to a weighted graph Laplacian with rotation-rotated right-hand
/// sides — plus the paper's motion prior (lambda_motion = 30, median-interval scaling).
pub fn optimize_pgo(factors: &[Factor], n_nodes: usize, n_iter: usize) -> Vec<Se3> {
    // ---- Initialization: chain averaged adjacent-frame measurements. ----
    let mut chain: Vec<Vec<Se3>> = vec![Vec::new(); n_nodes];
    for f in factors {
        if f.j == f.i + 1 {
            chain[f.j].push(f.z);
        }
    }
    let mut poses: Vec<Se3> = Vec::with_capacity(n_nodes);
    poses.push(Se3::IDENTITY);
    for i in 1..n_nodes {
        let refs = &chain[i];
        if refs.is_empty() {
            poses.push(poses[i - 1]);
            continue;
        }
        let mut t_avg = Vec3::zero();
        let mut q_acc = [0.0f32; 4];
        for z in refs {
            t_avg = t_avg.add(&z.t);
            let q = Quat::from_mat(&z.r);
            // Sign-aligned accumulation (mean, not product).
            let s = if q.0[0] < 0.0 { -1.0 } else { 1.0 };
            for c in 0..4 {
                q_acc[c] += s * q.0[c];
            }
        }
        let t_avg = t_avg.scale(1.0 / refs.len() as f32);
        let z = Se3 { r: Quat(q_acc).normalized().to_mat(), t: t_avg };
        // T_j = T_i * Z_ij (Z maps coordinates from j to i).
        poses.push(poses[i - 1].compose(&z));
    }

    // ---- Stage 1: rotations via iterative weighted geodesic averaging. ----
    // Per node, collect (weight, target rotation) from every incident factor:
    //   factor (i, j), i < j:  R_i^T R_j = R_ij  ->  R_i = R_j R_ij^T,  R_j = R_i R_ij.
    let mut incident: Vec<Vec<(usize, f32, Mat3, bool)>> = vec![Vec::new(); n_nodes];
    for f in factors {
        // For node i (as the earlier node): target R_i = R_j R_ij^T.
        incident[f.i].push((f.j, f.weight, poses[f.j].r.mul_mat(&f.z.r.transpose()), true));
        // For node j (as the later node): target R_j = R_i R_ij.
        incident[f.j].push((f.i, f.weight, poses[f.i].r.mul_mat(&f.z.r), false));
    }
    for _ in 0..n_iter.max(8) {
        let mut max_delta = 0.0f32;
        for i in 1..n_nodes {
            // Weighted quaternion mean of the targets (sign-aligned to the current
            // estimate so antipodal representations do not cancel).
            let mut acc = [0.0f32; 4];
            let mut wsum = 0.0f32;
            for &(_other, w, target, _is_i) in &incident[i] {
                let q = Quat::from_mat(&project_to_so3(&target));
                let ref_q = Quat::from_mat(&poses[i].r);
                let s = if q.sub_dot(&ref_q) < 0.0 { -1.0 } else { 1.0 };
                for c in 0..4 {
                    acc[c] += w * s * q.0[c];
                }
                wsum += w;
            }
            if wsum > 1e-9 {
                let q_new = Quat(acc).normalized();
                let q_old = Quat::from_mat(&poses[i].r);
                max_delta = max_delta.max((1.0 - q_new.sub_dot(&q_old).abs()).max(0.0));
                poses[i].r = q_new.to_mat();
            }
        }
        // Refresh targets from the updated rotations.
        for node in 0..n_nodes {
            for slot in 0..incident[node].len() {
                let (other, w, _, is_earlier) = incident[node][slot];
                // Find the factor direction again: we stored is_earlier = "node is i".
                let target = if is_earlier {
                    poses[other].r.mul_mat(&find_factor(factors, node, other).transpose())
                } else {
                    poses[other].r.mul_mat(&find_factor(factors, other, node))
                };
                incident[node][slot] = (other, w, target, is_earlier);
            }
        }
        if max_delta < 1e-6 {
            break;
        }
    }

    // ---- Stage 2: translations (Eq. 15 bottom) — linear LSQ + motion prior. ----
    {
        // Unknowns: t_i for nodes 1..n-1 (node 0 fixed at 0).
        let nu = (n_nodes - 1) * 3;
        let mut h = vec![0.0f32; nu * nu];
        let mut g = vec![0.0f32; nu];
        let omega_total: f32 = factors.iter().map(|f| f.weight).sum::<f32>().max(1e-9);
        for f in factors {
            let w = f.weight / omega_total;
            // Residual r = R_i^T (t_j - t_i) - t_ij; J_j = R_i^T, J_i = -R_i^T.
            // J^T J blocks reduce to +-w I (rotations cancel); J^T b keeps R_i t_ij.
            let ri_t = poses[f.i].r; // (R_i^T)^T = R_i
            let tij = f.z.t;
            for a in 0..3 {
                if f.j > 0 {
                    let jj = (f.j - 1) * 3;
                    h[(jj + a) * nu + (jj + a)] += w;
                }
                if f.i > 0 {
                    let ii = (f.i - 1) * 3;
                    h[(ii + a) * nu + (ii + a)] += w;
                    if f.j > 0 {
                        let jj = (f.j - 1) * 3;
                        h[(ii + a) * nu + (jj + a)] -= w;
                        h[(jj + a) * nu + (ii + a)] -= w;
                    }
                }
                // g = J^T b: node j gets +R_i t_ij, node i gets -R_i t_ij.
                let rot_t = ri_t.mul_vec(&tij);
                if f.j > 0 {
                    let jj = (f.j - 1) * 3;
                    g[jj + a] += w * rot_t.0[a];
                }
                if f.i > 0 {
                    let ii = (f.i - 1) * 3;
                    g[ii + a] -= w * rot_t.0[a];
                }
            }
        }
        // Motion prior: at interior nodes penalize (v_out - v_in) scaled by the median
        // sampling interval. With median_dt = dt the residual is the plain second
        // difference t_{i+1} - 2 t_i + t_{i-1}; the penalty is averaged over interior
        // nodes (paper Sec 5.2, lambda_motion = 30).
        let n_interior = (n_nodes.saturating_sub(2)).max(1);
        let k = LAMBDA_MOTION / n_interior as f32;
        for i in 1..(n_nodes - 1) {
            let blocks = [(i - 1, 1.0f32), (i, -2.0f32), (i + 1, 1.0f32)];
            for &(na, ca) in &blocks {
                if na == 0 {
                    continue;
                }
                for &(nb, cb) in &blocks {
                    if nb == 0 {
                        continue;
                    }
                    let ia = (na - 1) * 3;
                    let ib = (nb - 1) * 3;
                    for c in 0..3 {
                        h[(ia + c) * nu + (ib + c)] += k * ca * cb;
                    }
                }
            }
        }
        for i in 0..nu {
            h[i * nu + i] += 1e-6;
        }
        if let Some(tvec) = cholesky_solve(&h, &g, nu) {
            poses[0].t = Vec3::zero();
            for i in 1..n_nodes {
                poses[i].t = Vec3::new(tvec[(i - 1) * 3], tvec[(i - 1) * 3 + 1], tvec[(i - 1) * 3 + 2]);
            }
        }
    }
    poses
}

/// Fetch the measurement rotation for factor (i, j) — i must be the earlier node.
fn find_factor(factors: &[Factor], i: usize, j: usize) -> Mat3 {
    for f in factors {
        if f.i == i && f.j == j {
            return f.z.r;
        }
    }
    Mat3::IDENTITY
}

/// Project a near-rotation matrix back onto SO(3) (Gram-Schmidt on columns).
fn project_to_so3(m: &Mat3) -> Mat3 {
    let c0 = Vec3::new(m.0[0], m.0[3], m.0[6]).normalized();
    let c1 = Vec3::new(m.0[1], m.0[4], m.0[7]);
    let c1 = c1.sub(&c0.scale(c0.dot(&c1))).normalized();
    let c2 = c0.cross(&c1);
    Mat3([
        c0.x(), c1.x(), c2.x(),
        c0.y(), c1.y(), c2.y(),
        c0.z(), c1.z(), c2.z(),
    ])
}

/// Sequential window stitching (Eq. 14): align each window to the previous assembled
/// one with a single Sim(3) transform estimated by iteratively reweighted least
/// squares with a Huber penalty over depth-derived point correspondences.
/// Here correspondences are the per-frame rig origins + representative surface points.
pub fn stitch_sequential(windows: &[WindowPred]) -> Vec<Vec<Se3>> {
    // Eq. 14: align each new window to the already-assembled trajectory with a
    // single Sim(3) transform over the overlapping frames' predicted rig origins
    // and a representative scene point ahead of the rig.
    let mut out: Vec<Vec<Se3>> = Vec::with_capacity(windows.len());
    for (k, w) in windows.iter().enumerate() {
        if k == 0 {
            out.push(w.poses.clone());
            continue;
        }
        let prev = &out[k - 1];
        let mut xs: Vec<Vec3> = Vec::new();
        let mut ys: Vec<Vec3> = Vec::new();
        for (idx_new, &fi) in w.frames.iter().enumerate() {
            if let Some(idx_old) = windows[k - 1].frames.iter().position(|&f| f == fi) {
                // Rig origin.
                xs.push(w.poses[idx_new].t);
                ys.push(prev[idx_old].t);
                // Representative scene point 6 m ahead of the rig.
                xs.push(w.poses[idx_new].t.add(&w.poses[idx_new].r.mul_vec(&Vec3::new(6.0, 0.0, 0.0))));
                ys.push(prev[idx_old].t.add(&prev[idx_old].r.mul_vec(&Vec3::new(6.0, 0.0, 0.0))));
            }
        }
        if xs.len() < 3 {
            out.push(w.poses.clone());
            continue;
        }
        let sim = sim3_irls(&xs, &ys, 8);
        out.push(w.poses.iter().map(|p| sim.compose(p)).collect());
    }
    out
}

/// Estimate a Sim(3) transform (s, R, t) mapping x -> y by Umeyama init + IRLS with
/// Huber weights (Eq. 14).
pub fn sim3_irls(xs: &[Vec3], ys: &[Vec3], iters: usize) -> Se3 {
    // Umeyama (rotation + scale, then translation).
    let n = xs.len() as f32;
    let mut cx = Vec3::zero();
    let mut cy = Vec3::zero();
    for (x, y) in xs.iter().zip(ys.iter()) {
        cx = cx.add(x);
        cy = cy.add(y);
    }
    cx = cx.scale(1.0 / n);
    cy = cy.scale(1.0 / n);
    // Cross-covariance.
    let mut cov = [0.0f32; 9];
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dx = x.sub(&cx);
        let dy = y.sub(&cy);
        for i in 0..3 {
            for j in 0..3 {
                cov[i * 3 + j] += dy.0[i] * dx.0[j];
            }
        }
    }
    // SVD-free rotation via quaternion iteration (Horn's method).
    let r = horn_rotation(&cov);
    // Scale (we keep metric scale per the paper's PGO assumption; clamp to ~1).
    let mut var_x = 0.0f32;
    for x in xs.iter() {
        var_x += x.sub(&cx).norm2();
    }
    let mut tr = 0.0f32;
    for (x, y) in xs.iter().zip(ys.iter()) {
        let dy = y.sub(&cy);
        tr += r.mul_vec(&x.sub(&cx)).dot(&dy);
    }
    let s = if var_x > 1e-9 { (tr / var_x).clamp(0.9, 1.1) } else { 1.0 };
    let t = cy.sub(&r.mul_vec(&cx).scale(s));
    let mut pose = Se3 { r, t };
    // IRLS with Huber weights (delta = 0.5 m).
    let delta = 0.5f32;
    let mut weights = vec![1.0f32; xs.len()];
    for _ in 0..iters {
        // Recompute residuals and weights.
        let mut any = false;
        for (i, (x, y)) in xs.iter().zip(ys.iter()).enumerate() {
            let r_err = pose.apply(x).sub(y).norm();
            let w = if r_err <= delta { 1.0 } else { delta / r_err.max(1e-9) };
            if (w - weights[i]).abs() > 1e-6 {
                any = true;
            }
            weights[i] = w;
        }
        if !any {
            break;
        }
        // Weighted Umeyama.
        let wsum: f32 = weights.iter().sum();
        if wsum < 1e-6 {
            break;
        }
        let mut cx = Vec3::zero();
        let mut cy = Vec3::zero();
        for ((x, y), w) in xs.iter().zip(ys.iter()).zip(weights.iter()) {
            cx = cx.add(&x.scale(*w));
            cy = cy.add(&y.scale(*w));
        }
        cx = cx.scale(1.0 / wsum);
        cy = cy.scale(1.0 / wsum);
        let mut cov = [0.0f32; 9];
        for (((x, y), w), _) in xs.iter().zip(ys.iter()).zip(weights.iter()).zip(0..) {
            let dx = x.sub(&cx);
            let dy = y.sub(&cy);
            for i in 0..3 {
                for j in 0..3 {
                    cov[i * 3 + j] += w * dy.0[i] * dx.0[j];
                }
            }
        }
        let r = horn_rotation(&cov);
        let mut var_x = 0.0f32;
        for ((x, _), w) in xs.iter().zip(ys.iter()).zip(weights.iter()) {
            var_x += w * x.sub(&cx).norm2();
        }
        let mut tr = 0.0f32;
        for ((x, y), w) in xs.iter().zip(ys.iter()).zip(weights.iter()) {
            let dy = y.sub(&cy);
            tr += w * r.mul_vec(&x.sub(&cx)).dot(&dy);
        }
        let s = if var_x > 1e-9 { (tr / var_x).clamp(0.9, 1.1) } else { 1.0 };
        let t = cy.sub(&r.mul_vec(&cx).scale(s));
        pose = Se3 { r, t };
    }
    pose
}

/// Horn's closed-form rotation from the 3x3 cross-covariance A = sum dx dy^T:
/// build the symmetric 4x4 N matrix and extract its dominant eigenvector (the
/// rotation quaternion) by power iteration.
fn horn_rotation(h: &[f32; 9]) -> Mat3 {
    // A = H^T (we need sum dx dy^T; `h` is sum dy dx^T).
    let a = [
        h[0], h[3], h[6],
        h[1], h[4], h[7],
        h[2], h[5], h[8],
    ];
    let sxx = a[0];
    let sxy = a[1];
    let sxz = a[2];
    let syx = a[3];
    let syy = a[4];
    let syz = a[5];
    let szx = a[6];
    let szy = a[7];
    let szz = a[8];
    // Horn's N matrix (quaternion = [w, x, y, z]).
    let n = [
        sxx + syy + szz, syz - szy, szx - sxz, sxy - syx,
        syz - szy, sxx - syy - szz, sxy + syx, szx + sxz,
        szx - sxz, sxy + syx, -sxx + syy - szz, syz + szy,
        sxy - syx, szx + sxz, syz + szy, -sxx - syy + szz,
    ];
    // Jacobi eigendecomposition: the dominant eigenvector of N is the rotation
    // quaternion (exact for this 4x4 symmetric matrix, unlike power iteration whose
    // convergence rate depends on the eigenvalue gap).
    let (evals, evecs) = eig4_symmetric(&n);
    let mut best = 0usize;
    for c in 1..4 {
        if evals[c] > evals[best] {
            best = c;
        }
    }
    let q = [evecs[best], evecs[4 + best], evecs[8 + best], evecs[12 + best]];
    let q = if q[0] < 0.0 { [-q[0], -q[1], -q[2], -q[3]] } else { q };
    Quat(q).normalized().to_mat()
}

/// Jacobi eigendecomposition of a symmetric 4x4 matrix (row-major).
/// Returns (eigenvalues, eigenvector columns flattened row-major: evecs[row*4+col]).
fn eig4_symmetric(n: &[f32; 16]) -> ([f32; 4], [f32; 16]) {
    let mut a = *n;
    let mut v = [
        1.0, 0.0, 0.0, 0.0,
        0.0, 1.0, 0.0, 0.0,
        0.0, 0.0, 1.0, 0.0,
        0.0, 0.0, 0.0, 1.0,
    ];
    for _ in 0..32 {
        // Largest off-diagonal element.
        let (mut p, mut q) = (0usize, 1usize);
        let mut max = 0.0f32;
        for i in 0..4 {
            for j in (i + 1)..4 {
                if a[i * 4 + j].abs() > max {
                    max = a[i * 4 + j].abs();
                    p = i;
                    q = j;
                }
            }
        }
        if max < 1e-12 {
            break;
        }
        let app = a[p * 4 + p];
        let aqq = a[q * 4 + q];
        let apq = a[p * 4 + q];
        let theta = (aqq - app) / (2.0 * apq);
        let t = if theta >= 0.0 {
            1.0 / (theta + (1.0 + theta * theta).sqrt())
        } else {
            1.0 / (theta - (1.0 + theta * theta).sqrt())
        };
        let c = 1.0 / (1.0 + t * t).sqrt();
        let s = t * c;
        // A <- J^T A J (rows then columns p, q).
        for k in 0..4 {
            let akp = a[k * 4 + p];
            let akq = a[k * 4 + q];
            a[k * 4 + p] = c * akp - s * akq;
            a[k * 4 + q] = s * akp + c * akq;
        }
        for k in 0..4 {
            let apk = a[p * 4 + k];
            let aqk = a[q * 4 + k];
            a[p * 4 + k] = c * apk - s * aqk;
            a[q * 4 + k] = s * apk + c * aqk;
        }
        // V <- V J.
        for k in 0..4 {
            let vkp = v[k * 4 + p];
            let vkq = v[k * 4 + q];
            v[k * 4 + p] = c * vkp - s * vkq;
            v[k * 4 + q] = s * vkp + c * vkq;
        }
    }
    ([a[0], a[5], a[10], a[15]], v)
}

/// Trajectory RMSE vs ground truth after aligning the first pose (position error).
pub fn trajectory_rmse(est: &[Se3], gt: &[Se3]) -> f32 {
    let n = est.len().min(gt.len());
    if n == 0 {
        return 0.0;
    }
    // Rigid-align first pose to identity for both.
    let a0 = est[0].inverse();
    let g0 = gt[0].inverse();
    let mut sum = 0.0f32;
    for i in 0..n {
        let pe = a0.compose(&est[i]).t;
        let pg = g0.compose(&gt[i]).t;
        sum += pe.sub(&pg).norm2();
    }
    (sum / n as f32).sqrt()
}

/// Simulate per-window pose predictions (paper-stand-in for the rig/ego heads'
/// output quality): each window's inference is independent, with error growing
/// toward the window's temporal edges.
pub fn simulate_window_predictions(
    gt_ego: &[Se3],
    window_size: usize,
    stride: usize,
    rng: &mut Rng,
) -> Vec<WindowPred> {
    let n = gt_ego.len();
    let mut windows = Vec::new();
    let mut s = 0usize;
    while s + window_size <= n {
        let frames: Vec<usize> = (s..s + window_size).collect();
        // Each window's inference carries a small systematic drift (bias of the
        // feed-forward pose regression for that clip segment) plus iid noise that
        // grows toward the temporal edges — realistic window prediction quality.
        let drift_yaw = rng.normal_clamped(0.0012);
        let drift_t = Vec3::new(rng.normal_clamped(0.018), rng.normal_clamped(0.018), 0.0);
        // Predictions are expressed in the WINDOW frame (anchored at the window's
        // first reference frame, exactly as the paper formulates it): the anchor
        // frame is the reference by construction, and prediction error grows with
        // temporal distance from it.
        let anchor_inv = gt_ego[s].inverse();
        let poses: Vec<Se3> = frames
            .iter()
            .map(|&i| {
                let k = (i - s) as f32;
                if k == 0.0 {
                    return Se3::IDENTITY;
                }
                let sigma_r = 0.002 + 0.0015 * k;
                let sigma_t = 0.015 + 0.015 * k;
                let axis = Vec3::new(rng.normal_clamped(1.0), rng.normal_clamped(1.0), rng.normal_clamped(1.0)).normalized();
                let rr = Mat3::from_axis_angle(&axis, rng.normal_clamped(sigma_r) + k * drift_yaw);
                let tt = Vec3::new(
                    rng.normal_clamped(sigma_t) + k * drift_t.x(),
                    rng.normal_clamped(sigma_t) + k * drift_t.y(),
                    rng.normal_clamped(sigma_t * 0.4),
                );
                // Noise on the right (frame-local).
                let gt_rel = anchor_inv.compose(&gt_ego[i]);
                gt_rel.compose(&Se3 { r: rr, t: tt })
            })
            .collect();
        windows.push(WindowPred { frames, poses });
        s += stride;
    }
    windows
}

#[cfg(test)]
mod tests {
    use super::*;

    fn traj_line(n: usize, dx: f32, dyaw: f32) -> Vec<Se3> {
        let mut out = Vec::new();
        let mut yaw = 0.0f32;
        let mut p = Vec3::zero();
        for _ in 0..n {
            out.push(Se3 { r: Mat3::from_yaw(yaw), t: p });
            yaw += dyaw;
            p = p.add(&Vec3::new(yaw.cos() * dx, yaw.sin() * dx, 0.0));
        }
        out
    }

    #[test]
    fn pgo_reduces_noise() {
        let gt = traj_line(16, 3.2, 0.01);
        let mut rng = Rng::new(7);
        let windows = simulate_window_predictions(&gt, 4, 3, &mut rng);
        assert!(windows.len() >= 5);
        let factors = build_factors(&windows);
        assert!(!factors.is_empty());
        let opt = optimize_pgo(&factors, 16, 12);
        // Raw chaining error.
        let mut raw = vec![Se3::IDENTITY; 16];
        for w in &windows {
            for (k, &f) in w.frames.iter().enumerate() {
                if f == 0 {
                    raw[0] = w.poses[0];
                }
            }
        }
        // Anchor each estimate's first pose to GT for fair comparison.
        let rmse_raw = {
            // Chain from window 0 sequentially.
            let mut chained: Vec<Se3> = Vec::new();
            let mut cur = windows[0].poses[0];
            let anchor = cur.inverse();
            for w in &windows {
                let rel0 = w.poses[0].inverse();
                for p in &w.poses {
                    chained.push(anchor.compose(&rel0.compose(p)));
                }
            }
            // Take the last chain state per index.
            let mut per = vec![Se3::IDENTITY; 16];
            let mut k = 0usize;
            for w in &windows {
                for (idx, &f) in w.frames.iter().enumerate() {
                    per[f] = chained[k + idx];
                }
                k += w.poses.len();
            }
            trajectory_rmse(&per, &gt)
        };
        let rmse_pgo = trajectory_rmse(&opt, &gt);
        assert!(rmse_pgo < rmse_raw, "PGO {rmse_pgo} should beat raw {rmse_raw}");
        assert!(rmse_pgo < 0.6, "PGO RMSE {rmse_pgo}");
    }

    #[test]
    fn sim3_maps_points() {
        let r = Mat3::from_yaw(0.4);
        let t = Vec3::new(1.0, -2.0, 0.5);
        let xs: Vec<Vec3> = (0..20)
            .map(|i| Vec3::new(i as f32 * 0.7, (i % 5) as f32 * 1.3, (i % 3) as f32 * 0.8))
            .collect();
        let ys: Vec<Vec3> = xs.iter().map(|x| r.mul_vec(x).add(&t)).collect();
        let est = sim3_irls(&xs, &ys, 10);
        for (x, y) in xs.iter().zip(ys.iter()) {
            let p = est.apply(x);
            assert!(p.sub(y).norm() < 0.05, "sim3 err {}", p.sub(y).norm());
        }
    }
}
