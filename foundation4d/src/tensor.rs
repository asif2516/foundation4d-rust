//! SIMD-accelerated tensor kernels: GEMM variants, softmax, LayerNorm, GELU, RoPE,
//! and multi-head attention. Weights are stored "out-major" — `W[o*d_in + i]` — so the
//! inner product along the input channel axis is contiguous (SIMD-friendly).
//!
//! On wasm32 the hot loops use explicit 128-bit SIMD (`simd128` target feature,
//! supported by every browser since 2021); natively we rely on auto-vectorization.

#[cfg(target_arch = "wasm32")]
use core::arch::wasm32::*;

/// C[a, b] = sum_i A[a, i] * B[b, i]  (A: na x d, B: nb x d, C: na x nb).
/// This is the Q^T K / linear-layer core kernel.
#[allow(dead_code)]
pub fn gemm_at_b(c: &mut [f32], a: &[f32], b: &[f32], na: usize, nb: usize, d: usize) {
    #[cfg(target_arch = "wasm32")]
    {
        let d4 = d / 4 * 4;
        for ia in 0..na {
            let arow = &a[ia * d..(ia + 1) * d];
            let crow = &mut c[ia * nb..(ia + 1) * nb];
            for ib in 0..nb {
                let brow = &b[ib * d..(ib + 1) * d];
                let mut acc = f32x4_splat(0.0);
                unsafe {
                    let mut k = 0;
                    while k < d4 {
                        acc = f32x4_add(acc, f32x4_mul(v128_load(arow.as_ptr().add(k) as *const v128), v128_load(brow.as_ptr().add(k) as *const v128)));
                        k += 4;
                    }
                }
                let mut s = unsafe {
                    f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc)
                        + f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc)
                };
                let mut k = d4;
                while k < d {
                    s += arow[k] * brow[k];
                    k += 1;
                }
                crow[ib] = s;
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        for ia in 0..na {
            let arow = &a[ia * d..(ia + 1) * d];
            let crow = &mut c[ia * nb..(ia + 1) * nb];
            for ib in 0..nb {
                let brow = &b[ib * d..(ib + 1) * d];
                let mut s = 0.0f32;
                for k in (0..d).step_by(4) {
                    let rem = d - k;
                    if rem >= 4 {
                        s += arow[k] * brow[k] + arow[k + 1] * brow[k + 1]
                            + arow[k + 2] * brow[k + 2] + arow[k + 3] * brow[k + 3];
                    } else {
                        for j in 0..rem { s += arow[k + j] * brow[k + j]; }
                    }
                }
                crow[ib] = s;
            }
        }
    }
}

/// Linear layer: Y[n, o] = X[n, :] · W^T + bias, with W stored out-major (d_out x d_in).
pub fn linear(y: &mut [f32], x: &[f32], w: &[f32], b: &[f32], n: usize, d_in: usize, d_out: usize) {
    #[cfg(target_arch = "wasm32")]
    {
        let d4 = d_in / 4 * 4;
        for i in 0..n {
            let xrow = &x[i * d_in..(i + 1) * d_in];
            let yrow = &mut y[i * d_out..(i + 1) * d_out];
            for o in 0..d_out {
                let wrow = &w[o * d_in..(o + 1) * d_in];
                let mut acc = f32x4_splat(0.0);
                unsafe {
                    let mut k = 0;
                    while k < d4 {
                        acc = f32x4_add(acc, f32x4_mul(v128_load(xrow.as_ptr().add(k) as *const v128), v128_load(wrow.as_ptr().add(k) as *const v128)));
                        k += 4;
                    }
                }
                let mut s = unsafe {
                    f32x4_extract_lane::<0>(acc) + f32x4_extract_lane::<1>(acc)
                        + f32x4_extract_lane::<2>(acc) + f32x4_extract_lane::<3>(acc)
                };
                let mut k = d4;
                while k < d_in {
                    s += xrow[k] * wrow[k];
                    k += 1;
                }
                yrow[o] = s + b[o];
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..n {
            let xrow = &x[i * d_in..(i + 1) * d_in];
            let yrow = &mut y[i * d_out..(i + 1) * d_out];
            for o in 0..d_out {
                let wrow = &w[o * d_in..(o + 1) * d_in];
                let mut s = 0.0f32;
                for k in 0..d_in {
                    s += xrow[k] * wrow[k];
                }
                yrow[o] = s + b[o];
            }
        }
    }
}

/// C[n, d] = A[n, k] · B[k, d] (row-major matmul; used for scores · V).
pub fn gemm_ab(c: &mut [f32], a: &[f32], b: &[f32], n: usize, k: usize, d: usize) {
    #[cfg(target_arch = "wasm32")]
    {
        let d4 = d / 4 * 4;
        for i in 0..n {
            let arow = &a[i * k..(i + 1) * k];
            let crow = &mut c[i * d..(i + 1) * d];
            // Zero the row.
            for j in 0..d { crow[j] = 0.0; }
            for kk in 0..k {
                let av = arow[kk];
                if av == 0.0 { continue; }
                let brow = &b[kk * d..(kk + 1) * d];
                unsafe {
                    let mut j = 0;
                    while j < d4 {
                        let b4 = v128_load(brow.as_ptr().add(j) as *const v128);
                        let c4 = v128_load(crow.as_ptr().add(j) as *const v128);
                        v128_store(crow.as_mut_ptr().add(j) as *mut v128, f32x4_add(c4, f32x4_mul(b4, f32x4_splat(av))));
                        j += 4;
                    }
                }
                let mut j = d4;
                while j < d {
                    crow[j] += av * brow[j];
                    j += 1;
                }
            }
        }
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        for i in 0..n {
            let arow = &a[i * k..(i + 1) * k];
            let crow = &mut c[i * d..(i + 1) * d];
            for j in 0..d { crow[j] = 0.0; }
            for kk in 0..k {
                let av = arow[kk];
                let brow = &b[kk * d..(kk + 1) * d];
                for j in 0..d {
                    crow[j] += av * brow[j];
                }
            }
        }
    }
}

/// Row-wise in-place softmax over `n` rows of width `d`.
pub fn softmax_rows(x: &mut [f32], n: usize, d: usize) {
    for i in 0..n {
        let row = &mut x[i * d..(i + 1) * d];
        let mut mx = f32::NEG_INFINITY;
        for v in row.iter() { if *v > mx { mx = *v; } }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            *v = (*v - mx).exp();
            sum += *v;
        }
        let inv = 1.0 / sum.max(1e-12);
        for v in row.iter_mut() { *v *= inv; }
    }
}

/// LayerNorm over feature axis (in-place on `x`, n rows x d features).
pub fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], n: usize, d: usize, eps: f32) {
    for i in 0..n {
        let row = &mut x[i * d..(i + 1) * d];
        let mean = row.iter().sum::<f32>() / d as f32;
        let var = row.iter().map(|v| { let dv = v - mean; dv * dv }).sum::<f32>() / d as f32;
        let inv = 1.0 / (var + eps).sqrt();
        for j in 0..d {
            row[j] = (row[j] - mean) * inv * gamma[j] + beta[j];
        }
    }
}

/// In-place GELU (tanh approximation).
pub fn gelu(x: &mut [f32]) {
    for v in x.iter_mut() {
        let xv = *v;
        let t = (0.7978845608 * (xv + 0.044715 * xv * xv * xv)).tanh();
        *v = 0.5 * xv * (1.0 + t);
    }
}

/// Multi-head softmax attention. `q`, `k`, `v` are (n x d) row-major, `out` is (n x d).
/// Heads are contiguous channel slices. `scratch` must hold n * n floats (scores).
/// Per-head slices are compacted so the SIMD GEMM kernels can drive both the
/// Q^T K scores and the scores . V aggregation.
/// RoPE is applied by the caller to q/k before this call.
#[allow(clippy::too_many_arguments)]
pub fn attention_mh(
    out: &mut [f32],
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n: usize,
    d: usize,
    n_heads: usize,
    scratch: &mut [f32],
) {
    let dh = d / n_heads;
    let scale = 1.0 / (dh as f32).sqrt();
    let nd = n * d;
    // Compacted per-head buffers: [qh | kh | vh | oh], each n x dh.
    let mut comp = vec![0.0f32; 4 * n * dh];
    for h in 0..n_heads {
        let off = h * dh;
        let (qh, rest) = comp.split_at_mut(n * dh);
        let (kh, rest2) = rest.split_at_mut(n * dh);
        let (vh, oh) = rest2.split_at_mut(n * dh);
        for i in 0..n {
            qh[i * dh..(i + 1) * dh].copy_from_slice(&q[i * d + off..i * d + off + dh]);
            kh[i * dh..(i + 1) * dh].copy_from_slice(&k[i * d + off..i * d + off + dh]);
            vh[i * dh..(i + 1) * dh].copy_from_slice(&v[i * d + off..i * d + off + dh]);
        }
        // Scores (n x n) = qh . kh^T.
        gemm_at_b(scratch, qh, kh, n, n, dh);
        for s in scratch.iter_mut() {
            *s *= scale;
        }
        softmax_rows(scratch, n, n);
        // oh (n x dh) = scores . vh.
        gemm_ab(oh, scratch, vh, n, n, dh);
        // Scatter back.
        for i in 0..n {
            out[i * d + off..i * d + off + dh].copy_from_slice(&oh[i * dh..(i + 1) * dh]);
        }
    }
    let _ = nd;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gemm_at_b_correct() {
        let a = [1.0, 2.0, 3.0, 4.0]; // 2 x 2
        let b = [1.0, 0.0, 0.0, 1.0];
        let mut c = [0.0f32; 4];
        gemm_at_b(&mut c, &a, &b, 2, 2, 2);
        assert!((c[0] - 1.0).abs() < 1e-6); // (1,2)·(1,0)
        assert!((c[1] - 2.0).abs() < 1e-6);
        assert!((c[2] - 3.0).abs() < 1e-6);
        assert!((c[3] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn linear_correct() {
        let x = [1.0, 2.0]; // 1 x 2
        let w = [2.0, 0.0, 0.0, 3.0]; // out-major: 2 out x 2 in
        let b = [0.5, -0.5];
        let mut y = [0.0f32; 2];
        linear(&mut y, &x, &w, &b, 1, 2, 2);
        assert!((y[0] - 2.5).abs() < 1e-6);
        assert!((y[1] - 5.5).abs() < 1e-6);
    }

    #[test]
    fn gemm_ab_correct() {
        let a = [1.0, 2.0]; // 1 x 2
        let b = [1.0, 2.0, 3.0, 4.0]; // 2 x 2
        let mut c = [0.0f32; 2];
        gemm_ab(&mut c, &a, &b, 1, 2, 2);
        assert!((c[0] - 7.0).abs() < 1e-6);
        assert!((c[1] - 10.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut x = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        softmax_rows(&mut x, 2, 3);
        assert!((x[0] + x[1] + x[2] - 1.0).abs() < 1e-5);
        assert!(x[2] > x[1] && x[1] > x[0]);
    }

    #[test]
    fn attention_identity() {
        // Two tokens; token 0 query aligns with token 0 key (dot 1) vs token 1 (dot 0).
        // softmax([1, 0]) = [0.731, 0.269]. V rows: token0 = 0, token1 = 5.
        // out[token0] = 0.269 * 5 = 1.344 for every channel.
        let d = 4;
        let v = [0.0, 0.0, 0.0, 0.0, 5.0, 5.0, 5.0, 5.0];
        let mut out = vec![0.0f32; 8];
        let mut scratch = vec![0.0f32; 4];
        let q2 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        let k2 = [1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0];
        attention_mh(&mut out, &q2, &k2, &v, 2, d, 1, &mut scratch);
        // Scores are scaled by 1/sqrt(dh) = 0.5: softmax([0.5, 0]).
        let p10 = 1.0f32 / (1.0 + 0.5f32.exp()); // softmax([0.5,0])[1]
        assert!((out[0] - 5.0 * p10).abs() < 1e-4, "out0 {}", out[0]);
        assert!((out[1] - 5.0 * p10).abs() < 1e-4, "out1 {}", out[1]);
        // Token 1 (query aligns with token 1 key): softmax([0, 0.5]).
        let p11 = 0.5f32.exp() / (1.0 + 0.5f32.exp());
        assert!((out[4] - 5.0 * p11).abs() < 1e-4, "out4 {}", out[4]);
        assert!((out[5] - 5.0 * p11).abs() < 1e-4, "out5 {}", out[5]);
    }
}
