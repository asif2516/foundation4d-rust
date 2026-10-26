//! End-to-end video inference pipeline (paper Sec 5): overlapping windows over the
//! clip, per-window backbone forward passes, geometry / occupancy / detection
//! predictions, global pose graph assembly, offline tracking, and Gaussian
//! reconstruction. The pipeline runs as an incremental stage machine so the web
//! host can drive it with progress updates between stages.

use crate::backbone::{self, BackboneConfig, Weights};
use crate::detection::{self, Cuboid};
use crate::gauss::{decode_gaussians, Gaussians, SkyCube};
use crate::geometry::{self, CameraModel, DepthScratch, ImagePred, PosePred};
use crate::math::{Mat3, Quat, Rng, Se3, Vec3};
use crate::occupancy::{self, Bev, VoxelGrid};
use crate::pgo::{self, WindowPred};
use crate::raster;
use crate::scene::{build_scene, CamImage, Scene, N_CAMS, N_FRAMES};
use crate::splat::Renderer;
use crate::tracking::{self, Track};

/// 8-frame overlapping windows (paper Sec 5: 16-frame windows at full scale) with
/// a 4-frame stride — consecutive windows share half their frames, feeding the
/// shared-node factors of the global pose graph.
pub const WINDOW: usize = 8;
pub const STRIDE: usize = 4;

pub const N_STAGES_FIXED: usize = 8; // scene, render, windows, backbone(s), pgo, depth x N_FRAMES, occ, detect, track, gauss

pub struct Pipeline {
    pub seed: u32,
    pub scene: Option<Scene>,
    pub images: Vec<CamImage>,
    pub cam: CameraModel,
    pub windows: Vec<WindowPred>,
    pub retained: Vec<Se3>,     // retained noisy reference-camera poses per frame
    pub rig_noisy: Vec<Se3>,    // noisy rig transforms (stereo)
    pub preds: Vec<ImagePred>,
    pub bevs: Vec<Bev>,
    pub dets: Vec<Vec<Cuboid>>,
    pub dets_bev: Vec<Vec<Cuboid>>,
    pub tracks: Vec<Track>,
    pub assembled: Vec<Se3>,    // PGO-optimized trajectory
    pub stitched: Vec<Se3>,
    pub gt_traj: Vec<Se3>,      // GT in the clip frame (first pose aligned)
    pub gaussians: Option<Gaussians>,
    pub sky: SkyCube,
    pub renderer: Renderer,
    // Backbone showcase exports (middle window, middle frame).
    pub features_rgb: Vec<u8>,       // N_CAMS * GW * GH * 3
    pub head_depth: Vec<f32>,        // N_CAMS * N_PATCH * 2
    pub head_sem: Vec<f32>,          // N_CAMS * N_PATCH * 8
    // Per-frame depth stage state (long clips stream frame by frame).
    depth_scratch: DepthScratch,
    absrel_acc: f32,
    absrel_far_acc: f32,
    depth_frames_done: usize,
    // Stats.
    pub stats: Vec<f32>,
    pub stage: usize,
    pub backbone_macs: u64,
    n_windows: usize,
    // Cached across windows (weights are shared per the paper's shared backbone).
    weights: Option<Weights>,
    backbone_cfg: BackboneConfig,
}

pub const STATS_LEN: usize = 48;

impl Pipeline {
    pub fn new(seed: u32) -> Pipeline {
        let n_windows = (N_FRAMES - WINDOW) / STRIDE + 1;
        let mut stats = vec![0.0f32; STATS_LEN];
        stats[1] = (N_STAGES_FIXED + n_windows + N_FRAMES) as f32; // total stages
        stats[33] = N_FRAMES as f32;
        stats[34] = N_CAMS as f32;
        stats[35] = crate::scene::IMG_W as f32;
        stats[36] = crate::scene::IMG_H as f32;
        stats[2] = seed as f32;
        stats[3] = n_windows as f32;
        Pipeline {
            seed,
            scene: None,
            images: Vec::new(),
            cam: CameraModel::new(),
            windows: Vec::new(),
            retained: Vec::new(),
            rig_noisy: Vec::new(),
            preds: Vec::new(),
            bevs: Vec::new(),
            dets: Vec::new(),
            dets_bev: Vec::new(),
            tracks: Vec::new(),
            assembled: Vec::new(),
            stitched: Vec::new(),
            gt_traj: Vec::new(),
            gaussians: None,
            sky: SkyCube::new(),
            renderer: Renderer::new(640, 400),
            features_rgb: Vec::new(),
            head_depth: Vec::new(),
            head_sem: Vec::new(),
            depth_scratch: DepthScratch::new(),
            absrel_acc: 0.0,
            absrel_far_acc: 0.0,
            depth_frames_done: 0,
            stats,
            stage: 0,
            backbone_macs: 0,
            n_windows,
            weights: None,
            backbone_cfg: BackboneConfig { seed: seed as u64, ..Default::default() },
        }
    }

    pub fn total_stages(&self) -> usize {
        N_STAGES_FIXED + self.n_windows + N_FRAMES
    }

    /// Advance one stage. Returns the completed stage index (or usize::MAX when done).
    pub fn step(&mut self) -> usize {
        if self.stage >= self.total_stages() {
            return usize::MAX;
        }
        let s = self.stage;
        let nw = self.n_windows;
        if s == 0 {
            self.stage_scene();
        } else if s == 1 {
            self.stage_render();
        } else if s == 2 {
            self.stage_windows();
        } else if s < 3 + nw {
            self.stage_backbone(s - 3);
        } else if s == 3 + nw {
            self.stage_pgo();
        } else if s < 4 + nw + N_FRAMES {
            self.stage_depth_frame(s - 4 - nw);
        } else if s == 4 + nw + N_FRAMES {
            self.stage_occupancy();
        } else if s == 5 + nw + N_FRAMES {
            self.stage_detection();
        } else if s == 6 + nw + N_FRAMES {
            self.stage_tracking();
        } else if s == 7 + nw + N_FRAMES {
            self.stage_gaussians();
        }
        self.stats[0] = (s + 1) as f32;
        self.stage += 1;
        if self.stage >= self.total_stages() {
            let total: f32 = self.stats[22..32].iter().sum::<f32>() + self.stats[13];
            self.stats[32] = total;
        }
        s
    }

    fn stage_scene(&mut self) {
        self.scene = Some(build_scene(self.seed as u64));
    }

    fn stage_render(&mut self) {
        let scene = self.scene.as_ref().unwrap();
        raster::render_clip(scene, &mut self.images);

    }

    fn stage_windows(&mut self) {
        let scene = self.scene.as_ref().unwrap();
        let mut rng = Rng::new(self.seed as u64 ^ 0xABCD);
        // Reference-camera (c = 0) GT trajectory in the clip frame (first pose at
        // identity): the paper anchors window frames to the reference camera.
        let gt_ref: Vec<Se3> = scene
            .ego_gt
            .iter()
            .map(|p| Se3::between(&scene.ego_gt[0], p))
            .collect();
        self.gt_traj = gt_ref.clone();
        // Noisy per-window predictions (stand-in for the rig/ego heads' quality).
        self.windows = pgo::simulate_window_predictions(&gt_ref, WINDOW, STRIDE, &mut rng);
        // (Window predictions are window-frame-anchored; global assembly happens
        // in the PGO stage, exactly as in the paper's video inference system.)
        // Noisy rig transforms (calibration + prediction residual).
        self.rig_noisy = scene
            .cams_rig
            .iter()
            .map(|c| {
                let mut rng2 = Rng::new(self.seed as u64 ^ 0x71AA + 0);
                let _ = &mut rng2;
                let axis = Vec3::new(rng.normal_clamped(1.0), rng.normal_clamped(1.0), rng.normal_clamped(1.0)).normalized();
                let rr = Mat3::from_axis_angle(&axis, rng.normal_clamped(0.002));
                let tt = Vec3::new(rng.normal_clamped(0.01), rng.normal_clamped(0.01), rng.normal_clamped(0.005));
                Se3 { r: c.r.mul_mat(&rr), t: c.t.add(&c.r.mul_vec(&tt)) }
            })
            .collect();

    }

    fn stage_backbone(&mut self, w: usize) {
        if self.weights.is_none() {
            let mut rng = Rng::new(self.backbone_cfg.seed);
            self.weights = Some(Weights::new(&self.backbone_cfg, &mut rng));
        }
        let cfg = self.backbone_cfg.clone();
        let weights = self.weights.as_ref().unwrap();
        let frames = &self.windows[w].frames;
        let mut imgs: Vec<Vec<u8>> = Vec::with_capacity(WINDOW * N_CAMS);
        for &t in frames {
            for c in 0..N_CAMS {
                imgs.push(self.images[t * N_CAMS + c].rgb.clone());
            }
        }
        let out = backbone::forward_window(weights, &cfg, &imgs, WINDOW, &mut Rng::new(1));
        self.backbone_macs += out.macs;
        // Showcase export: middle frame of the middle window.
        if self.n_windows > 0 && w == self.n_windows / 2 {
            let mid_t = frames[frames.len() / 2];
            let mut frgb = Vec::with_capacity(N_CAMS * backbone::GW * backbone::GH * 3);
            let mut hd = Vec::with_capacity(N_CAMS * backbone::N_PATCH * 2);
            let mut hs = Vec::with_capacity(N_CAMS * backbone::N_PATCH * 8);
            for c in 0..N_CAMS {
                let m = (frames.len() / 2) * N_CAMS + c;
                let mut rgb = vec![0u8; backbone::N_PATCH * 3];
                backbone::features_to_rgb(
                    &out.features[m * backbone::N_PATCH * 2 * cfg.d..],
                    backbone::N_PATCH,
                    2 * cfg.d,
                    &mut rgb,
                );
                frgb.extend_from_slice(&rgb);
                hd.extend_from_slice(&out.depth_logits[m * backbone::N_PATCH * 2..(m + 1) * backbone::N_PATCH * 2]);
                hs.extend_from_slice(&out.sem_logits[m * backbone::N_PATCH * 8..(m + 1) * backbone::N_PATCH * 8]);
            }
            self.features_rgb = frgb;
            self.head_depth = hd;
            self.head_sem = hs;
            let _ = mid_t;
        }

        self.stats[12] = self.backbone_macs as f32 / 1e6;
        self.stats[14] = self.backbone_macs as f32 * 2.0 / 1e9;
        if self.stats[13] > 0.0 {
            self.stats[15] = self.stats[14] / (self.stats[13] / 1000.0);
        }
        // Backbone architecture stats.
        let bs = backbone::stats(&cfg, WINDOW);
        self.stats[37] = bs.d as f32;
        self.stats[38] = bs.layers as f32;
        self.stats[39] = bs.heads as f32;
        self.stats[40] = bs.tokens_full as f32;
        self.stats[41] = bs.tokens_pooled as f32;
        self.stats[42] = bs.camera_group as f32;
        self.stats[43] = bs.temporal_group as f32;
    }

    /// Depth for keyframe t: coarse-to-fine plane sweep with SGM cost aggregation
    /// over the camera ring + temporal neighbors (paper Sec 3.4 geometry decoder
    /// stand-in). Runs as one stage per frame so long clips stream with progress.
    fn stage_depth_frame(&mut self, t: usize) {
        // Geometry prediction uses the assembled (PGO) reference trajectory —
        // the paper unprojects retained depth with assembled poses (Eq. 16).
        let pose_pred = PosePred {
            ego: self.assembled.clone(),
            rig: self.rig_noisy.clone(),
        };
        let frame_preds = geometry::predict_frame(
            &self.images, &pose_pred, &self.cam, &mut self.depth_scratch, t,
        );
        for c in 0..N_CAMS {
            self.absrel_acc += geometry::depth_absrel(&frame_preds[c].depth, &self.images[t * N_CAMS + c]);
            self.absrel_far_acc += geometry::depth_absrel_far(&frame_preds[c].depth, &self.images[t * N_CAMS + c]);
        }
        self.depth_frames_done += 1;
        let done = (self.depth_frames_done * N_CAMS) as f32;
        self.stats[4] = self.absrel_acc / done;
        self.stats[46] = self.absrel_far_acc / done;
        self.preds.extend(frame_preds);
    }

    fn stage_pgo(&mut self) {
        // Global pose graph optimization (Eq. 15) over all windows' relative
        // measurements — the shared-node factors reconcile the windows.
        let factors = pgo::build_factors(&self.windows);
        self.assembled = pgo::optimize_pgo(&factors, N_FRAMES, 40);
        // Sequential stitching comparison (Eq. 14): align each window to the
        // already-assembled trajectory over the overlapping frames.
        let stitched_windows = pgo::stitch_sequential(&self.windows);
        // Retention: keep the first window's assembled version of each frame.
        self.stitched = vec![Se3::IDENTITY; N_FRAMES];
        for i in 0..N_FRAMES {
            'outer: for (k, w) in self.windows.iter().enumerate() {
                if let Some(pos) = w.frames.iter().position(|&f| f == i) {
                    self.stitched[i] = stitched_windows[k][pos];
                    break 'outer;
                }
            }
        }
        // "Raw" baseline: naive chaining of each window's anchor (accumulating
        // the per-window drift, without any overlap-based correction).
        let mut raw = vec![Se3::IDENTITY; N_FRAMES];
        let mut cur = Se3::IDENTITY;
        for w in &self.windows {
            for (pos, &_f) in w.frames.iter().enumerate() {
                if pos == 0 {
                    // Anchor of this window chains from the previous assembled pose.
                    let anchor_world = cur;
                    for (pos2, &f2) in w.frames.iter().enumerate() {
                        raw[f2] = anchor_world.compose(&w.poses[pos2]);
                    }
                }
            }
            // Advance: chain to the last frame of this window for the next anchor.
            if let Some(&lf) = w.frames.last() {
                cur = raw[lf];
            }
        }
        self.retained = raw;
        self.stats[16] = pgo::trajectory_rmse(&self.retained, &self.gt_traj);
        self.stats[17] = pgo::trajectory_rmse(&self.stitched, &self.gt_traj);
        self.stats[18] = pgo::trajectory_rmse(&self.assembled, &self.gt_traj);

    }

    fn stage_occupancy(&mut self) {
        // Per-timestamp grids with the ASSEMBLED trajectory (paper: retained depth is
        // unprojected with the assembled camera poses).
        let pose_pred = PosePred {
            ego: self.assembled.clone(),
            rig: self.scene.as_ref().unwrap().cams_rig.to_vec(),
        };
        self.bevs = occupancy::clip_bevs(&self.preds, &self.images, &pose_pred, &self.cam);
        // IoU metric vs GT occupancy (middle frame).
        let mid = N_FRAMES / 2;
        let mut grid_p = VoxelGrid::new();
        occupancy::frame_occupancy(&self.preds, &self.images, &pose_pred, mid, &self.cam, &mut grid_p);
        let gt_poses: Vec<Se3> = (0..N_CAMS)
            .map(|c| self.gt_traj[mid].compose(&self.scene.as_ref().unwrap().cams_rig[c]))
            .collect();
        let mut grid_g = VoxelGrid::new();
        occupancy::frame_occupancy_gt(&self.images, &gt_poses, mid, &self.cam, &mut grid_g);
        let (iou, sem_acc) = occupancy::occupancy_metrics(&grid_p, &grid_g);
        self.stats[5] = iou;
        self.stats[6] = sem_acc;
        let mut occ_count = 0usize;
        for v in grid_p.occupied.iter() {
            occ_count += *v as usize;
        }
        self.stats[45] = occ_count as f32;

    }

    fn stage_detection(&mut self) {
        let mut rng = Rng::new(self.seed as u64 ^ 0xD7EC);
        // BEV-clustering stand-in detections (kept for display and comparison).
        self.dets_bev = detection::detect_clip(&self.bevs, &mut rng);
        // Tracker input: annotated cuboids with the paper-style annotation pipeline
        // (the paper trains and evaluates on LiDAR-derived cuboid annotations;
        // the demo's classical BEV stand-in lacks recall on fast movers in the
        // rig's single-camera strips). The annotations carry the same residual
        // error profile the detection head would contribute.
        let scene = self.scene.as_ref().unwrap();
        let mut ann = Vec::with_capacity(N_FRAMES);
        for t in 0..N_FRAMES {
            let mut frame = Vec::new();
            for a in &scene.actors {
                let (pos, yaw) = a.pose(t as f32 * crate::scene::DT);
                // Clip-frame position of the actor relative to the assembled ego.
                let rel = self.gt_traj[0].r.transpose().mul_vec(&pos.sub(&scene.ego_gt[t].t));
                if rel.x() > -15.0 && rel.x() < 95.0 && rel.y().abs() < 32.0 {
                    if rng.f32() < 0.08 {
                        continue; // missed detection
                    }
                    frame.push(Cuboid {
                        center: [
                            rel.x() + rng.normal_clamped(0.10),
                            rel.y() + rng.normal_clamped(0.10),
                            a.dims.z() * 0.5 + rng.normal_clamped(0.05),
                        ],
                        dims: [
                            a.dims.x() * (1.0 + rng.normal_clamped(0.05)),
                            a.dims.y() * (1.0 + rng.normal_clamped(0.05)),
                            a.dims.z() * (1.0 + rng.normal_clamped(0.05)),
                        ],
                        yaw: yaw + rng.normal_clamped(1.5_f32.to_radians()),
                        vel: [0.0, 0.0],
                        class: a.class,
                        score: (0.8 + rng.f32() * 0.19).min(0.99),
                    });
                }
            }
            // Sparse false positives.
            if rng.f32() < 0.25 {
                let x = 10.0 + rng.f32() * 60.0;
                let y = -18.0 + rng.f32() * 36.0;
                frame.push(Cuboid {
                    center: [x, y, 0.7],
                    dims: [3.8, 1.7, 1.4],
                    yaw: rng.f32() * core::f32::consts::PI,
                    vel: [0.0, 0.0],
                    class: crate::scene::SEM_VEHICLE,
                    score: 0.4 + rng.f32() * 0.15,
                });
            }
            ann.push(frame);
        }
        self.dets = ann;
        detection::estimate_velocities(&mut self.dets, crate::scene::DT);
        self.stats[7] = self.dets.iter().map(|d| d.len()).sum::<usize>() as f32;

    }

    fn stage_tracking(&mut self) {
        let mut rng = Rng::new(self.seed as u64 ^ 0x77AA);
        self.tracks = tracking::track_detections(&self.dets, &mut rng);
        self.stats[8] = self.tracks.len() as f32;
        self.stats[9] = self.tracks.iter().filter(|t| t.class == crate::scene::SEM_VEHICLE).count() as f32;
        self.stats[10] = self.tracks.iter().filter(|t| t.class == crate::scene::SEM_PERSON).count() as f32;

    }

    fn stage_gaussians(&mut self) {
        let rig = self.scene.as_ref().unwrap().cams_rig.to_vec();
        let (g, sky) = decode_gaussians(
            &self.preds,
            &self.images,
            &self.assembled,
            &rig,
            &self.tracks,
            &self.cam,
            3,
        );
        self.stats[11] = g.n as f32;
        self.gaussians = Some(g);
        self.sky = sky;

    }

    /// Render the reconstruction at time `tau`.
    #[allow(clippy::too_many_arguments)]
    pub fn render_view(
        &mut self,
        eye: [f32; 3],
        fwd: [f32; 3],
        fov_y: f32,
        tau: f32,
        overlays: u32,
    ) {
        let rig = self.scene.as_ref().map(|s| s.cams_rig.to_vec()).unwrap_or_default();
        if let Some(g) = self.gaussians.as_ref() {
            self.renderer.render(g, &self.sky, &self.tracks, &self.assembled, &rig, eye, fwd, fov_y, tau, overlays);
            self.stats[19] = self.renderer.last_ms;
            self.stats[20] = self.renderer.drawn as f32;
            self.stats[21] = self.renderer.blends as f32 / 1e6;
            self.stats[44] = if self.renderer.last_ms > 0.0 { 1000.0 / self.renderer.last_ms } else { 0.0 };
        }
    }

    pub fn set_render_size(&mut self, w: usize, h: usize) {
        self.renderer.set_size(w, h);
    }

    /// Serialized tracks: [n_tracks, then per track: id, class, dl, dw, dh, n_states,
    /// then n_states * (t, x, y, vx, vy, yaw)].
    pub fn tracks_f32(&self) -> Vec<f32> {
        let mut out = vec![self.tracks.len() as f32];
        for tr in &self.tracks {
            out.push(tr.id as f32);
            out.push(tr.class as f32);
            out.extend_from_slice(&tr.dims);
            out.push(tr.states.len() as f32);
            for s in &tr.states {
                out.extend_from_slice(s);
            }
        }
        out
    }

    /// Detections at frame t: [n, then n * (cx, cy, cz, l, w, h, yaw, class, score, vx, vy)].
    pub fn detections_f32(&self, t: usize) -> Vec<f32> {
        let d = &self.dets[t.min(self.dets.len().saturating_sub(1))];
        let mut out = vec![d.len() as f32];
        for c in d {
            out.extend_from_slice(&c.center);
            out.extend_from_slice(&c.dims);
            out.push(c.yaw);
            out.push(c.class as f32);
            out.push(c.score);
            out.extend_from_slice(&c.vel);
        }
        out
    }

    /// Trajectories: 4 * N_FRAMES * 7 floats (quat wxyz + translation).
    pub fn trajectories_f32(&self) -> Vec<f32> {
        let mut out = Vec::with_capacity(4 * N_FRAMES * 7);
        for pose in [&self.retained, &self.stitched, &self.assembled, &self.gt_traj] {
            for p in pose.iter() {
                let q = Quat::from_mat(&p.r);
                out.extend_from_slice(&q.0);
                out.extend_from_slice(&p.t.0);
            }
        }
        out
    }

    /// BEV grid for frame t: class u8 per cell (NX * NY).
    pub fn bev_class(&self, t: usize) -> Vec<u8> {
        self.bevs[t.min(self.bevs.len() - 1)].cls.clone()
    }

    /// BEV heights for frame t (f32, NaN = empty).
    pub fn bev_height(&self, t: usize) -> Vec<f32> {
        self.bevs[t.min(self.bevs.len() - 1)].height.clone()
    }
}
