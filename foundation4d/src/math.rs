//! Minimal linear algebra for Foundation4D: Vec3, Mat3, unit quaternions, SE(3) poses.
//! Everything is `f32`, stack-friendly, and allocation-free in hot paths.

/// 3-vector.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Vec3(pub [f32; 3]);

impl Vec3 {
    #[inline]
    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Vec3([x, y, z])
    }
    #[inline]
    pub fn zero() -> Self {
        Vec3([0.0; 3])
    }
    #[inline]
    pub fn x(&self) -> f32 { self.0[0] }
    #[inline]
    pub fn y(&self) -> f32 { self.0[1] }
    #[inline]
    pub fn z(&self) -> f32 { self.0[2] }
    #[inline]
    pub fn add(&self, o: &Vec3) -> Vec3 {
        Vec3([self.0[0] + o.0[0], self.0[1] + o.0[1], self.0[2] + o.0[2]])
    }
    #[inline]
    pub fn sub(&self, o: &Vec3) -> Vec3 {
        Vec3([self.0[0] - o.0[0], self.0[1] - o.0[1], self.0[2] - o.0[2]])
    }
    #[inline]
    pub fn scale(&self, s: f32) -> Vec3 {
        Vec3([self.0[0] * s, self.0[1] * s, self.0[2] * s])
    }
    #[inline]
    pub fn dot(&self, o: &Vec3) -> f32 {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2]
    }
    #[inline]
    pub fn cross(&self, o: &Vec3) -> Vec3 {
        Vec3([
            self.0[1] * o.0[2] - self.0[2] * o.0[1],
            self.0[2] * o.0[0] - self.0[0] * o.0[2],
            self.0[0] * o.0[1] - self.0[1] * o.0[0],
        ])
    }
    #[inline]
    pub fn norm(&self) -> f32 {
        self.dot(self).sqrt()
    }
    #[inline]
    pub fn norm2(&self) -> f32 {
        self.dot(self)
    }
    #[inline]
    pub fn normalized(&self) -> Vec3 {
        let n = self.norm();
        if n < 1e-12 { Vec3::new(0.0, 0.0, 1.0) } else { self.scale(1.0 / n) }
    }
}

/// 3x3 row-major matrix.
#[derive(Clone, Copy, Debug)]
pub struct Mat3(pub [f32; 9]);

impl Mat3 {
    pub const IDENTITY: Mat3 = Mat3([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);

    #[inline]
    pub fn mul_vec(&self, v: &Vec3) -> Vec3 {
        let m = &self.0;
        Vec3([
            m[0] * v.0[0] + m[1] * v.0[1] + m[2] * v.0[2],
            m[3] * v.0[0] + m[4] * v.0[1] + m[5] * v.0[2],
            m[6] * v.0[0] + m[7] * v.0[1] + m[8] * v.0[2],
        ])
    }
    #[inline]
    pub fn mul_mat(&self, o: &Mat3) -> Mat3 {
        let a = &self.0;
        let b = &o.0;
        let mut r = [0.0f32; 9];
        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a[i * 3 + k] * b[k * 3 + j];
                }
                r[i * 3 + j] = s;
            }
        }
        Mat3(r)
    }
    pub fn transpose(&self) -> Mat3 {
        let m = &self.0;
        Mat3([
            m[0], m[3], m[6],
            m[1], m[4], m[7],
            m[2], m[5], m[8],
        ])
    }
    pub fn trace(&self) -> f32 {
        self.0[0] + self.0[4] + self.0[8]
    }
    /// Frobenius norm.
    pub fn frob_norm(&self) -> f32 {
        self.0.iter().map(|v| v * v).sum::<f32>().sqrt()
    }
    pub fn frob_norm2(&self) -> f32 {
        self.0.iter().map(|v| v * v).sum::<f32>()
    }
    /// Rotation matrix from axis-angle (axis must be unit; angle in radians).
    pub fn from_axis_angle(axis: &Vec3, angle: f32) -> Mat3 {
        let (s, c) = angle.sin_cos();
        let one_c = 1.0 - c;
        let (x, y, z) = (axis.x(), axis.y(), axis.z());
        Mat3([
            c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s,
            y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s,
            z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c,
        ])
    }
    /// Rotation matrix from yaw (around +z), used for rig cameras / object yaw.
    pub fn from_yaw(a: f32) -> Mat3 {
        let (s, c) = a.sin_cos();
        Mat3([c, -s, 0.0, s, c, 0.0, 0.0, 0.0, 1.0])
    }
    /// Rotation around +x (pitch).
    pub fn from_pitch(a: f32) -> Mat3 {
        let (s, c) = a.sin_cos();
        Mat3([1.0, 0.0, 0.0, 0.0, c, -s, 0.0, s, c])
    }
    /// Matrix logarithm: axis * angle from a rotation matrix.
    pub fn log(&self) -> Vec3 {
        // Clamp trace into valid cosine range.
        let c = ((self.trace() - 1.0) * 0.5).clamp(-1.0, 1.0);
        let angle = c.acos();
        if angle.abs() < 1e-6 {
            // Small angle: antisymmetric part / 2.
            return Vec3::new(
                0.5 * (self.0[7] - self.0[5]),
                0.5 * (self.0[2] - self.0[6]),
                0.5 * (self.0[3] - self.0[1]),
            );
        }
        let k = angle / (2.0 * angle.sin());
        Vec3::new(
            k * (self.0[7] - self.0[5]),
            k * (self.0[2] - self.0[6]),
            k * (self.0[3] - self.0[1]),
        )
    }
}

/// Unit quaternion (w, x, y, z).
#[derive(Clone, Copy, Debug)]
pub struct Quat(pub [f32; 4]);

impl Quat {
    pub const IDENTITY: Quat = Quat([1.0, 0.0, 0.0, 0.0]);

    pub fn from_mat(m: &Mat3) -> Quat {
        // Standard branch-stable extraction (Shepperd / Wikipedia formulation).
        let tr = m.trace();
        let (w, x, y, z);
        if tr > 0.0 {
            let s = (tr + 1.0).sqrt() * 2.0; // s = 4w
            w = 0.25 * s;
            x = (m.0[7] - m.0[5]) / s;
            y = (m.0[2] - m.0[6]) / s;
            z = (m.0[3] - m.0[1]) / s;
        } else if m.0[0] > m.0[4] && m.0[0] > m.0[8] {
            let s = (1.0 + m.0[0] - m.0[4] - m.0[8]).sqrt() * 2.0; // s = 4x
            w = (m.0[7] - m.0[5]) / s;
            x = 0.25 * s;
            y = (m.0[1] + m.0[3]) / s;
            z = (m.0[2] + m.0[6]) / s;
        } else if m.0[4] > m.0[8] {
            let s = (1.0 + m.0[4] - m.0[0] - m.0[8]).sqrt() * 2.0; // s = 4y
            w = (m.0[2] - m.0[6]) / s;
            x = (m.0[1] + m.0[3]) / s;
            y = 0.25 * s;
            z = (m.0[5] + m.0[7]) / s;
        } else {
            let s = (1.0 + m.0[8] - m.0[0] - m.0[4]).sqrt() * 2.0; // s = 4z
            w = (m.0[3] - m.0[1]) / s;
            x = (m.0[2] + m.0[6]) / s;
            y = (m.0[5] + m.0[7]) / s;
            z = 0.25 * s;
        }
        // Canonical form: non-negative scalar part.
        if w < 0.0 {
            Quat([-w, -x, -y, -z]).normalized()
        } else {
            Quat([w, x, y, z]).normalized()
        }
    }

    pub fn from_axis_angle(axis: &Vec3, angle: f32) -> Quat {
        let h = angle * 0.5;
        let s = h.sin();
        let a = axis.normalized();
        Quat([h.cos(), a.x() * s, a.y() * s, a.z() * s])
    }

    #[inline]
    pub fn normalized(&self) -> Quat {
        let n = self.0.iter().map(|v| v * v).sum::<f32>().sqrt();
        if n < 1e-12 { Quat::IDENTITY } else {
            Quat([self.0[0] / n, self.0[1] / n, self.0[2] / n, self.0[3] / n])
        }
    }

    pub fn to_mat(&self) -> Mat3 {
        let [w, x, y, z] = self.0;
        Mat3([
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
            2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
        ])
    }

    pub fn conjugate(&self) -> Quat {
        Quat([self.0[0], -self.0[1], -self.0[2], -self.0[3]])
    }

    /// Dot product with another quaternion.
    pub fn sub_dot(&self, o: &Quat) -> f32 {
        self.0[0] * o.0[0] + self.0[1] * o.0[1] + self.0[2] * o.0[2] + self.0[3] * o.0[3]
    }

    pub fn mul(&self, o: &Quat) -> Quat {
        let [w1, x1, y1, z1] = self.0;
        let [w2, x2, y2, z2] = o.0;
        Quat([
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ])
    }

    pub fn rotate(&self, v: &Vec3) -> Vec3 {
        self.to_mat().mul_vec(v)
    }

    /// Shortest-arc spherical interpolation (paper Sec 3.4: ego trajectory interpolation
    /// uses linear interpolation for translation and slerp for rotation).
    pub fn slerp(&self, other: &Quat, t: f32) -> Quat {
        let mut d = self.0[0] * other.0[0] + self.0[1] * other.0[1]
            + self.0[2] * other.0[2] + self.0[3] * other.0[3];
        let mut q2 = *other;
        if d < 0.0 {
            q2 = Quat([-other.0[0], -other.0[1], -other.0[2], -other.0[3]]);
            d = -d;
        }
        if d > 0.9995 {
            // Linear interpolation, then normalize.
            let r = Quat([
                self.0[0] + t * (q2.0[0] - self.0[0]),
                self.0[1] + t * (q2.0[1] - self.0[1]),
                self.0[2] + t * (q2.0[2] - self.0[2]),
                self.0[3] + t * (q2.0[3] - self.0[3]),
            ]);
            return r.normalized();
        }
        let theta0 = d.acos();
        let theta = theta0 * t;
        let st0 = (theta0).sin();
        let st = (theta).sin();
        let s0 = (theta0 - theta).sin() / st0;
        let s1 = st / st0;
        Quat([
            s0 * self.0[0] + s1 * q2.0[0],
            s0 * self.0[1] + s1 * q2.0[1],
            s0 * self.0[2] + s1 * q2.0[2],
            s0 * self.0[3] + s1 * q2.0[3],
        ]).normalized()
    }
}

/// SE(3) rigid transform: camera-to-world style (rotation + translation).
#[derive(Clone, Copy, Debug)]
pub struct Se3 {
    pub r: Mat3,
    pub t: Vec3,
}

impl Se3 {
    pub const IDENTITY: Se3 = Se3 { r: Mat3::IDENTITY, t: Vec3([0.0, 0.0, 0.0]) };

    pub fn from_quat_t(q: &Quat, t: &Vec3) -> Se3 {
        Se3 { r: q.to_mat(), t: *t }
    }

    /// Compose: (self ∘ other) — apply `other` first, then `self`.
    pub fn compose(&self, other: &Se3) -> Se3 {
        Se3 { r: self.r.mul_mat(&other.r), t: self.r.mul_vec(&other.t).add(&self.t) }
    }

    pub fn inverse(&self) -> Se3 {
        let ri = self.r.transpose();
        Se3 { r: ri, t: ri.mul_vec(&self.t).scale(-1.0) }
    }

    pub fn apply(&self, p: &Vec3) -> Vec3 {
        self.r.mul_vec(p).add(&self.t)
    }

    /// Relative transform mapping `b`'s coordinates into `a`'s frame: a^-1 ∘ b.
    pub fn between(a: &Se3, b: &Se3) -> Se3 {
        a.inverse().compose(b)
    }

    /// Interpolate two poses: lerp translation + slerp rotation (paper Sec 3.4).
    pub fn interp(a: &Se3, b: &Se3, t: f32) -> Se3 {
        let qa = Quat::from_mat(&a.r);
        let qb = Quat::from_mat(&b.r);
        let q = qa.slerp(&qb, t);
        let tr = a.t.add(&b.t.sub(&a.t).scale(t));
        Se3 { r: q.to_mat(), t: tr }
    }
}

/// Dense Cholesky solver for small SPD systems (n <= ~200). Returns None if not SPD.
pub fn cholesky_solve(a: &[f32], b: &[f32], n: usize) -> Option<Vec<f32>> {
    // In-place lower-triangular factorization of `a` (row-major n x n).
    let mut l = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut s = a[i * n + j];
            for k in 0..j {
                s -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if s <= 1e-10 { return None; }
                l[i * n + j] = s.sqrt();
            } else {
                l[i * n + j] = s / l[j * n + j];
            }
        }
    }
    // Forward substitution: L y = b.
    let mut y = vec![0.0f32; n];
    for i in 0..n {
        let mut s = b[i];
        for k in 0..i {
            s -= l[i * n + k] * y[k];
        }
        y[i] = s / l[i * n + i];
    }
    // Back substitution: L^T x = y.
    let mut x = vec![0.0f32; n];
    for i in (0..n).rev() {
        let mut s = y[i];
        for k in (i + 1)..n {
            s -= l[k * n + i] * x[k];
        }
        x[i] = s / l[i * n + i];
    }
    Some(x)
}

/// Small PRNG (xorshift64*) — deterministic weights across platforms.
pub struct Rng(pub u64);

impl Rng {
    pub fn new(seed: u64) -> Rng {
        Rng(seed.wrapping_mul(0x9E3779B97F4A7C15).max(1))
    }
    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545F4914F6CDD1D)
    }
    #[inline]
    pub fn f32(&mut self) -> f32 {
        ((self.next_u64() >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Standard normal via Box-Muller.
    pub fn normal(&mut self) -> f32 {
        let u1 = self.f32().max(1e-7);
        let u2 = self.f32();
        (-2.0 * u1.ln()).sqrt() * (2.0 * core::f32::consts::PI * u2).sin()
    }
    /// Truncated-ish normal clamped to +-3 sigma.
    pub fn normal_clamped(&mut self, sigma: f32) -> f32 {
        self.normal().clamp(-3.0, 3.0) * sigma
    }
}

/// Deterministic hash noise in [0,1) from 3D quantized position — view-independent
/// surface texture so multi-view stereo has matchable high-frequency detail.
#[inline]
pub fn hash3(x: f32, y: f32, z: f32) -> f32 {
    let mut h: u32 = 2166136261;
    for v in [x.to_bits(), y.to_bits(), z.to_bits()] {
        h ^= v;
        h = h.wrapping_mul(16777619);
    }
    h ^= h >> 15;
    h = h.wrapping_mul(2246822519);
    h ^= h >> 13;
    (h >> 8) as f32 / 16777216.0
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: f32, b: f32, eps: f32, msg: &str) {
        assert!((a - b).abs() < eps, "{msg}: {a} vs {b}");
    }

    #[test]
    fn quat_roundtrip() {
        let q = Quat::from_axis_angle(&Vec3::new(1.0, 2.0, -0.5).normalized(), 0.83);
        let m = q.to_mat();
        let q2 = Quat::from_mat(&m);
        for i in 0..4 {
            assert_close(q.0[i], q2.0[i], 1e-4, "quat roundtrip");
        }
    }

    #[test]
    fn slerp_endpoints() {
        let a = Quat::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), 0.2);
        let b = Quat::from_axis_angle(&Vec3::new(0.0, 0.0, 1.0), 1.2);
        let m = a.slerp(&b, 0.0).to_mat();
        assert_close(m.0[2], a.to_mat().0[2], 1e-5, "slerp t=0");
        let m = a.slerp(&b, 1.0).to_mat();
        assert_close(m.0[2], b.to_mat().0[2], 1e-5, "slerp t=1");
        let mid = a.slerp(&b, 0.5);
        assert_close(mid.0[0], (0.35f32).cos(), 1e-2, "slerp mid");
    }

    #[test]
    fn se3_compose_inverse() {
        let a = Se3::from_quat_t(&Quat::from_axis_angle(&Vec3::new(0.3, 1.0, -0.2).normalized(), 0.5), &Vec3::new(1.0, -2.0, 0.5));
        let b = Se3::from_quat_t(&Quat::from_axis_angle(&Vec3::new(-0.7, 0.1, 1.0).normalized(), -1.1), &Vec3::new(-3.0, 0.0, 2.0));
        let c = a.compose(&b);
        let p = Vec3::new(4.0, 5.0, -6.0);
        let pc = c.apply(&p);
        let pb = a.apply(&b.apply(&p));
        for i in 0..3 {
            assert_close(pc.0[i], pb.0[i], 1e-4, "compose associativity");
        }
        let ci = c.inverse();
        let back = ci.apply(&pc);
        for i in 0..3 {
            assert_close(back.0[i], p.0[i], 1e-4, "inverse");
        }
    }

    #[test]
    fn cholesky() {
        // A = [[4,1],[1,3]], b = [1,2] -> x = [1/11, 7/11]... solve directly.
        let a = [4.0, 1.0, 1.0, 3.0];
        let b = [1.0, 2.0];
        let x = cholesky_solve(&a, &b, 2).unwrap();
        assert_close(4.0 * x[0] + 1.0 * x[1], 1.0, 1e-5, "chol row1");
        assert_close(1.0 * x[0] + 3.0 * x[1], 2.0, 1e-5, "chol row2");
    }

    #[test]
    fn mat_log_exp() {
        let axis = Vec3::new(0.2, -1.0, 0.4).normalized();
        let m = Mat3::from_axis_angle(&axis, 0.77);
        let lv = m.log();
        assert_close(lv.norm(), 0.77, 1e-4, "log norm");
        let m2 = Mat3::from_axis_angle(&lv.normalized(), lv.norm());
        for i in 0..9 {
            assert_close(m.0[i], m2.0[i], 1e-4, "log/exp roundtrip");
        }
    }
}
