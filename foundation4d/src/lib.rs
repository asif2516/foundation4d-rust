//! Foundation4D (arXiv 2610.30340v1) — unified perception & reconstruction for
//! autonomous vehicles, implemented in Rust.
//!
//! This crate implements the paper's inference stack end-to-end at demo scale,
//! with no external dependencies:
//!   * `backbone`  — shared encoder + factorized spatiotemporal aggregator
//!                   (within-image / cross-camera / cross-temporal attention with
//!                   rig, ego and register tokens — paper Sec 3.3);
//!   * `geometry`  — pose factorization T = T_ego(tau) * T_rig (Eq. 1), plane-sweep
//!                   depth (stand-in for the trained DPT geometry decoder),
//!                   pointmap unprojection (Eq. 2);
//!   * `occupancy` — semantic voxel grid fused from multi-view depth (paper Sec 3.6);
//!   * `detection` — 3D cuboids from BEV occupancy + velocity estimation (Sec 3.5);
//!   * `tracking`  — constant-velocity Kalman filter, Hungarian association,
//!                   two-phase offline MOT with an RTS smoother (Sec 5.3);
//!   * `pgo`       — global pose graph optimization with motion prior (Eq. 15),
//!                   sequential window stitching (Eq. 14);
//!   * `gauss`     — per-pixel 3D Gaussians from depth + assembled poses (Eq. 9)
//!                   with actor association and a sky cubemap (Sec 3.7);
//!   * `splat`     — tile-based Gaussian rasterizer for novel-view rendering;
//!   * `pipeline`  — overlapping-window video inference orchestration (Sec 5).
//!
//! The WASM ABI (compiled for wasm32 with `simd128`) exposes a flat C interface
//! consumed by the web demo.

pub mod math;
pub mod tensor;
pub mod scene;
pub mod raster;
pub mod backbone;
pub mod geometry;
pub mod occupancy;
pub mod detection;
pub mod tracking;
pub mod pgo;
pub mod gauss;
pub mod splat;
pub mod pipeline;

// ---------------------------------------------------------------------------
// Flat C ABI for the wasm32 cdylib (consumed by the web demo's JS glue).
// All heavy state lives behind a thread-local pipeline; pointers returned point
// into WASM linear memory and stay valid until the next mutating call.
// ---------------------------------------------------------------------------

#[cfg(target_arch = "wasm32")]
mod abi {
    use super::math::Quat;
    use super::pipeline::{Pipeline, STATS_LEN};
    use super::scene::{IMG_H, IMG_W, N_CAMS, N_FRAMES};
    use super::occupancy::{NX, NY};
    use core::cell::RefCell;

    thread_local! {
        static PIPE: RefCell<Option<Pipeline>> = RefCell::new(None);
    }

    fn with_pipe<R>(f: impl FnOnce(&Pipeline) -> R) -> R {
        PIPE.with_borrow(|p| f(p.as_ref().expect("pipeline not initialized")))
    }
    fn with_pipe_mut<R>(f: impl FnOnce(&mut Pipeline) -> R) -> R {
        PIPE.with_borrow_mut(|p| f(p.as_mut().expect("pipeline not initialized")))
    }

    /// Null-safe empty-slice pointer.
    fn ptr_u8(v: &[u8]) -> *const u8 {
        if v.is_empty() {
            core::ptr::null()
        } else {
            v.as_ptr()
        }
    }
    fn ptr_f32(v: &[f32]) -> *const f32 {
        if v.is_empty() {
            core::ptr::null()
        } else {
            v.as_ptr()
        }
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_version() -> u32 {
        2
    }

    /// JS-driven timing: the host measures each stage / render call and reports
    /// the duration here (the wasm32-unknown-unknown target has no clock).
    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_set_stat(idx: u32, v: f32) {
        with_pipe_mut(|p| {
            if (idx as usize) < p.stats.len() {
                p.stats[idx as usize] = v;
            }
        });
    }

    /// Record a render-frame duration measured by the host; also refreshes the
    /// splat statistics (drawn tile entries, blend operations, FPS).
    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_note_render(ms: f32) {
        with_pipe_mut(|p| {
            p.stats[19] = ms;
            p.stats[20] = p.renderer.drawn as f32;
            p.stats[21] = p.renderer.blends as f32 / 1e6;
            p.stats[44] = if ms > 0.0 { 1000.0 / ms } else { 0.0 };
        });
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_init(seed: u32) -> i32 {
        PIPE.with_borrow_mut(|p| {
            *p = Some(Pipeline::new(seed));
        });
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_total_stages() -> i32 {
        with_pipe(|p| p.total_stages() as i32)
    }

    /// Advance one pipeline stage; returns the completed stage index, -1 when done.
    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_step() -> i32 {
        with_pipe_mut(|p| match p.step() {
            usize::MAX => -1,
            s => s as i32,
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_stats_ptr() -> *const f32 {
        with_pipe(|p| ptr_f32(&p.stats))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_stats_len() -> u32 {
        STATS_LEN as u32
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_image_ptr(t: u32, c: u32) -> *const u8 {
        let (t, c) = (t as usize, c as usize);
        if t >= N_FRAMES || c >= N_CAMS {
            return core::ptr::null();
        }
        with_pipe(|p| ptr_u8(&p.images[t * N_CAMS + c].rgb))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_gt_depth_ptr(t: u32, c: u32) -> *const f32 {
        let (t, c) = (t as usize, c as usize);
        if t >= N_FRAMES || c >= N_CAMS {
            return core::ptr::null();
        }
        with_pipe(|p| ptr_f32(&p.images[t * N_CAMS + c].depth))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_gt_sem_ptr(t: u32, c: u32) -> *const u8 {
        let (t, c) = (t as usize, c as usize);
        if t >= N_FRAMES || c >= N_CAMS {
            return core::ptr::null();
        }
        with_pipe(|p| ptr_u8(&p.images[t * N_CAMS + c].sem))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_pred_depth_ptr(t: u32, c: u32) -> *const f32 {
        let (t, c) = (t as usize, c as usize);
        if t >= N_FRAMES || c >= N_CAMS || p_is_empty_pred(t, c) {
            return core::ptr::null();
        }
        with_pipe(|p| ptr_f32(&p.preds[t * N_CAMS + c].depth))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_pred_conf_ptr(t: u32, c: u32) -> *const f32 {
        let (t, c) = (t as usize, c as usize);
        if t >= N_FRAMES || c >= N_CAMS || p_is_empty_pred(t, c) {
            return core::ptr::null();
        }
        with_pipe(|p| ptr_f32(&p.preds[t * N_CAMS + c].conf))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_pred_sem_ptr(t: u32, c: u32) -> *const u8 {
        let (t, c) = (t as usize, c as usize);
        if t >= N_FRAMES || c >= N_CAMS || p_is_empty_pred(t, c) {
            return core::ptr::null();
        }
        with_pipe(|p| ptr_u8(&p.preds[t * N_CAMS + c].sem))
    }

    fn p_is_empty_pred(t: usize, c: usize) -> bool {
        with_pipe(|p| p.preds.len() <= t * N_CAMS + c)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_features_ptr() -> *const u8 {
        with_pipe(|p| ptr_u8(&p.features_rgb))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_features_len() -> u32 {
        (N_CAMS * (IMG_W / 8) * (IMG_H / 8) * 3) as u32
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_head_depth_ptr() -> *const f32 {
        with_pipe(|p| ptr_f32(&p.head_depth))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_head_sem_ptr() -> *const f32 {
        with_pipe(|p| ptr_f32(&p.head_sem))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_bev_class_ptr(t: u32) -> *const u8 {
        let t = t as usize;
        if with_pipe(|p| p.bevs.len()) <= t {
            return core::ptr::null();
        }
        with_pipe(|p| ptr_u8(&p.bevs[t].cls))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_bev_height_ptr(t: u32) -> *const f32 {
        let t = t as usize;
        if with_pipe(|p| p.bevs.len()) <= t {
            return core::ptr::null();
        }
        with_pipe(|p| ptr_f32(&p.bevs[t].height))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_bev_dims() -> u32 {
        ((NX as u32) << 16) | NY as u32
    }

    // Track / detection buffers are re-serialized per request into thread-local
    // staging so their pointers stay stable between reads.
    thread_local! {
        static STAGE_BUF: RefCell<Vec<f32>> = RefCell::new(Vec::new());
        static STAGE_U8: RefCell<Vec<u8>> = RefCell::new(Vec::new());
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_tracks_ptr() -> *const f32 {
        let data = with_pipe(|p| p.tracks_f32());
        STAGE_BUF.with_borrow_mut(|b| {
            *b = data;
            ptr_f32(b)
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_tracks_len() -> u32 {
        with_pipe(|p| p.tracks_f32().len() as u32)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_detections_ptr(t: u32) -> *const f32 {
        let t = t as usize;
        let data = with_pipe(|p| p.detections_f32(t));
        STAGE_BUF.with_borrow_mut(|b| {
            *b = data;
            ptr_f32(b)
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_detections_len(t: u32) -> u32 {
        with_pipe(|p| p.detections_f32(t as usize).len() as u32)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_traj_ptr() -> *const f32 {
        let data = with_pipe(|p| p.trajectories_f32());
        STAGE_BUF.with_borrow_mut(|b| {
            *b = data;
            ptr_f32(b)
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_gaussian_count() -> u32 {
        with_pipe(|p| p.gaussians.as_ref().map(|g| g.n as u32).unwrap_or(0))
    }

    // ---- Gaussian SoA exports (WebGPU splat renderer path). ----
    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_gauss_pos_ptr() -> *const f32 {
        with_pipe(|p| match p.gaussians.as_ref() {
            Some(g) if g.n > 0 => g.pos.as_ptr(),
            _ => core::ptr::null(),
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_gauss_quat_ptr() -> *const f32 {
        with_pipe(|p| match p.gaussians.as_ref() {
            Some(g) if g.n > 0 => g.quat.as_ptr(),
            _ => core::ptr::null(),
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_gauss_scale_ptr() -> *const f32 {
        with_pipe(|p| match p.gaussians.as_ref() {
            Some(g) if g.n > 0 => g.scale.as_ptr(),
            _ => core::ptr::null(),
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_gauss_color_ptr() -> *const u8 {
        with_pipe(|p| match p.gaussians.as_ref() {
            Some(g) if g.n > 0 => g.color.as_ptr(),
            _ => core::ptr::null(),
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_gauss_opacity_ptr() -> *const f32 {
        with_pipe(|p| match p.gaussians.as_ref() {
            Some(g) if g.n > 0 => g.opacity.as_ptr(),
            _ => core::ptr::null(),
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_gauss_actor_ptr() -> *const i32 {
        with_pipe(|p| match p.gaussians.as_ref() {
            Some(g) if g.n > 0 => g.actor.as_ptr(),
            _ => core::ptr::null(),
        })
    }

    /// Sky cubemap: 6 faces x 16x16 RGB (+X, -X, +Y, -Y, +Z, -Z).
    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_sky_ptr() -> *const u8 {
        with_pipe(|p| p.sky.faces.as_ptr())
    }

    /// Rig transforms (camera c in rig frame), 6 x (quat wxyz + t): 42 floats.
    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_rig_ptr() -> *const f32 {
        let mut rig = Vec::with_capacity(N_CAMS * 7);
        with_pipe(|p| {
            if let Some(s) = p.scene.as_ref() {
                for c in 0..N_CAMS {
                    let q = Quat::from_mat(&s.cams_rig[c].r);
                    rig.extend_from_slice(&q.0);
                    rig.extend_from_slice(&s.cams_rig[c].t.0);
                }
            }
        });
        STAGE_BUF.with_borrow_mut(|b| {
            *b = rig;
            ptr_f32(b)
        })
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_set_render_size(w: u32, h: u32) -> i32 {
        with_pipe_mut(|p| p.set_render_size(w.max(16) as usize, h.max(16) as usize));
        0
    }

    /// Render the reconstruction. Returns 0 on success, -1 if not ready.
    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_render(
        ex: f32, ey: f32, ez: f32,
        fx: f32, fy: f32, fz: f32,
        fov: f32, tau: f32, overlays: u32,
    ) -> i32 {
        let ready = with_pipe(|p| p.gaussians.is_some());
        if !ready {
            return -1;
        }
        with_pipe_mut(|p| {
            p.render_view([ex, ey, ez], [fx, fy, fz], fov, tau, overlays);
        });
        0
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_render_ptr() -> *const u8 {
        with_pipe(|p| ptr_u8(&p.renderer.rgba))
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_render_w() -> i32 {
        with_pipe(|p| p.renderer.w as i32)
    }

    #[unsafe(no_mangle)]
    pub extern "C" fn f4d_render_h() -> i32 {
        with_pipe(|p| p.renderer.h as i32)
    }
}

#[cfg(target_arch = "wasm32")]
pub use abi::*;
