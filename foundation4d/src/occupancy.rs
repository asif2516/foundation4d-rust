//! Semantic occupancy (paper Sec 3.6): a metric voxel grid in the clip frame,
//! fused by ray-marching the multi-view depth predictions — the same
//! LiDAR-aggregation principle the paper uses for its occupancy labels.
//! The grid covers the paper's driving ROI ([-20,100] x [-30,30] x [-3,5] m)
//! at 0.5 m voxels (demo scale; the paper uses 0.2 m).
//! Detections and the BEV map are derived from the per-timestamp grids.

use crate::geometry::{ImagePred, PosePred};
use crate::math::Se3;
use crate::raster::SKY_DEPTH;
use crate::scene::{CamImage, N_CAMS, N_FRAMES, SEM_SKY, N_SEM, IMG_H, IMG_W};

pub const VOXEL: f32 = 0.5;
pub const X_MIN: f32 = -20.0;
pub const X_MAX: f32 = 100.0;
pub const Y_MIN: f32 = -30.0;
pub const Y_MAX: f32 = 30.0;
pub const Z_MIN: f32 = -3.0;
pub const Z_MAX: f32 = 5.0;
pub const NX: usize = ((X_MAX - X_MIN) / VOXEL) as usize; // 240
pub const NY: usize = ((Y_MAX - Y_MIN) / VOXEL) as usize; // 120
pub const NZ: usize = ((Z_MAX - Z_MIN) / VOXEL) as usize; // 16

/// Per-voxel accumulator: hit counts, semantic votes, and mean observed color
/// (color supports cluster-level classification in the detection stage).
pub struct VoxelGrid {
    pub hits: Vec<u16>,
    pub sem_votes: Vec<[u8; N_SEM]>,
    pub col_r: Vec<u32>,
    pub col_g: Vec<u32>,
    pub col_b: Vec<u32>,
    pub col_n: Vec<u32>,
    pub cams: Vec<u8>, // bit i set: camera i observed this voxel
    pub occupied: Vec<u8>, // final: 1 = occupied
    pub sem: Vec<u8>,      // final: argmax semantic class
}

impl VoxelGrid {
    pub fn new() -> VoxelGrid {
        VoxelGrid {
            hits: vec![0; NX * NY * NZ],
            sem_votes: vec![[0; N_SEM]; NX * NY * NZ],
            col_r: vec![0; NX * NY * NZ],
            col_g: vec![0; NX * NY * NZ],
            col_b: vec![0; NX * NY * NZ],
            col_n: vec![0; NX * NY * NZ],
            cams: vec![0; NX * NY * NZ],
            occupied: vec![0; NX * NY * NZ],
            sem: vec![0; NX * NY * NZ],
        }
    }

    pub fn clear(&mut self) {
        self.hits.iter_mut().for_each(|v| *v = 0);
        self.sem_votes.iter_mut().for_each(|v| *v = [0; N_SEM]);
        self.col_r.iter_mut().for_each(|v| *v = 0);
        self.col_g.iter_mut().for_each(|v| *v = 0);
        self.col_b.iter_mut().for_each(|v| *v = 0);
        self.col_n.iter_mut().for_each(|v| *v = 0);
        self.cams.iter_mut().for_each(|v| *v = 0);
    }

    #[inline]
    fn idx(x: usize, y: usize, z: usize) -> usize {
        (z * NY + y) * NX + x
    }

    /// Ray-march one depth map into the grid (subsampled pixels, fixed step).
    /// `rgb` supplies the source pixel color for per-voxel color accumulation.
    #[allow(clippy::too_many_arguments)]
    pub fn integrate(
        &mut self,
        pred: &ImagePred,
        rgb: &[u8],
        pose: &Se3,
        cam: &crate::geometry::CameraModel,
        cam_idx: usize,
        stride: usize,
        min_hits: u16,
    ) {
        let step = VOXEL * 0.8;
        for py in (0..IMG_H).step_by(stride) {
            for px in (0..IMG_W).step_by(stride) {
                let p = py * IMG_W + px;
                let d = pred.depth[p];
                if d >= SKY_DEPTH * 0.9 {
                    continue; // sky contributes nothing
                }
                let sem = pred.sem[p];
                let conf = pred.conf[p];
                if conf < 0.02 {
                    continue;
                }
                let ray_cam = &cam.rays[p];
                let dir = pose.r.mul_vec(ray_cam);
                let dist = d / ray_cam.z();
                // March from near the camera to the endpoint, marking free space,
                // then register a hit with a semantic vote at the endpoint.
                let mut s = 0.6f32;
                while s < dist - VOXEL {
                    let w = pose.t.add(&dir.scale(s));
                    let (gx, gy, gz) = (
                        ((w.x() - X_MIN) / VOXEL) as isize,
                        ((w.y() - Y_MIN) / VOXEL) as isize,
                        ((w.z() - Z_MIN) / VOXEL) as isize,
                    );
                    if gx >= 0 && gx < NX as isize && gy >= 0 && gy < NY as isize && gz >= 0 && gz < NZ as isize {
                        // Free-space evidence: clear weak hits along the ray.
                        let i = Self::idx(gx as usize, gy as usize, gz as usize);
                        if self.hits[i] < min_hits {
                            self.hits[i] = 0;
                            self.sem_votes[i] = [0; N_SEM];
                        }
                    }
                    s += step;
                }
                // Endpoint hit at two ray distances bracketing the surface (a
                // half-voxel of depth tolerance — occupancy benchmarks routinely
                // evaluate with one-voxel dilation).
                let (pr, pg, pb) = (rgb[p * 3] as u32, rgb[p * 3 + 1] as u32, rgb[p * 3 + 2] as u32);
                for &d_end in &[dist + 0.15, dist - 0.35] {
                    let w = pose.t.add(&dir.scale(d_end));
                    let (gx, gy, gz) = (
                        ((w.x() - X_MIN) / VOXEL) as isize,
                        ((w.y() - Y_MIN) / VOXEL) as isize,
                        ((w.z() - Z_MIN) / VOXEL) as isize,
                    );
                    if gx >= 0 && gx < NX as isize && gy >= 0 && gy < NY as isize && gz >= 0 && gz < NZ as isize {
                        let i = Self::idx(gx as usize, gy as usize, gz as usize);
                        self.hits[i] = self.hits[i].saturating_add(1);
                        self.sem_votes[i][sem as usize] = self.sem_votes[i][sem as usize].saturating_add(2);
                        self.col_r[i] += pr;
                        self.col_g[i] += pg;
                        self.col_b[i] += pb;
                        self.col_n[i] += 1;
                        self.cams[i] |= 1u8 << (cam_idx & 7);
                    }
                }
            }
        }
    }

    /// Finalize occupancy: a voxel is occupied when enough rays terminated in it.
    /// Elevated voxels (obstacles) require stronger support than ground voxels —
    /// single-view depth outliers otherwise create floating phantom obstacles.
    pub fn finalize(&mut self, min_hits: u16) {
        for i in 0..NX * NY * NZ {
            let z = i / (NX * NY); // voxel layer index
            let elevated = z as f32 * VOXEL + Z_MIN > 0.4;
            let need = if elevated { min_hits + 1 } else { min_hits };
            if self.hits[i] >= need {
                self.occupied[i] = 1;
                let mut best = 0usize;
                let mut best_v = 0u8;
                for c in 0..N_SEM {
                    if self.sem_votes[i][c] > best_v {
                        best_v = self.sem_votes[i][c];
                        best = c;
                    }
                }
                self.sem[i] = best as u8;
            } else {
                self.occupied[i] = 0;
            }
        }
    }
}

/// BEV summary of a voxel grid: per-cell obstacle height (above the ground band),
/// mean obstacle color, and dominant class. "Obstacle" = occupied voxels above the
/// ground band that are not road / sidewalk / sky — trees and buildings included;
/// the detection stage classifies clusters by size, shape and color.
pub struct Bev {
    pub height: Vec<f32>,      // topmost occupied z per cell (NaN if empty)
    pub obst_height: Vec<f32>, // topmost obstacle z per cell (NaN if none)
    pub cls: Vec<u8>,          // dominant class among occupied voxels in the column
    pub any_obstacle: Vec<u8>, // 1 if any obstacle voxel in the column
    pub mean_rgb: Vec<f32>,    // 3 per cell: mean obstacle color (0-255)
}

pub fn bev_summary(grid: &VoxelGrid) -> Bev {
    let mut height = vec![f32::NEG_INFINITY; NX * NY];
    let mut obst_height = vec![f32::NEG_INFINITY; NX * NY];
    let mut cls = vec![0u8; NX * NY];
    let mut any_obstacle = vec![0u8; NX * NY];
    let mut mean_rgb = vec![0.0f32; NX * NY * 3];
    let mut votes = vec![[0u16; N_SEM]; NX * NY];
    let mut rs = vec![0u64; NX * NY];
    let mut gs = vec![0u64; NX * NY];
    let mut bs = vec![0u64; NX * NY];
    let mut ns = vec![0u64; NX * NY];
    let ground_classes = [
        crate::scene::SEM_ROAD as usize,
        crate::scene::SEM_SIDEWALK as usize,
        crate::scene::SEM_SKY as usize,
    ];
    for z in 0..NZ {
        let wz = Z_MIN + z as f32 * VOXEL;
        for y in 0..NY {
            for x in 0..NX {
                let i = (z * NY + y) * NX + x;
                if grid.occupied[i] == 0 {
                    continue;
                }
                let b = y * NX + x;
                let c = grid.sem[i] as usize;
                votes[b][c] += 1;
                if wz > height[b] {
                    height[b] = wz;
                }
                // Obstacle: above the ground band and not a ground class.
                if wz > 0.4 && !ground_classes.contains(&c) {
                    if wz > obst_height[b] {
                        obst_height[b] = wz;
                    }
                    any_obstacle[b] = 1;
                    if grid.col_n[i] > 0 {
                        rs[b] += grid.col_r[i] as u64;
                        gs[b] += grid.col_g[i] as u64;
                        bs[b] += grid.col_b[i] as u64;
                        ns[b] += grid.col_n[i] as u64;
                    }
                }
            }
        }
    }
    for b in 0..NX * NY {
        let mut best = 0usize;
        let mut best_v = 0u16;
        for c in 0..N_SEM {
            if votes[b][c] > best_v {
                best_v = votes[b][c];
                best = c;
            }
        }
        cls[b] = best as u8;
        if height[b] == f32::NEG_INFINITY {
            height[b] = f32::NAN;
        }
        if obst_height[b] == f32::NEG_INFINITY {
            obst_height[b] = f32::NAN;
        }
        if ns[b] > 0 {
            mean_rgb[b * 3] = rs[b] as f32 / ns[b] as f32;
            mean_rgb[b * 3 + 1] = gs[b] as f32 / ns[b] as f32;
            mean_rgb[b * 3 + 2] = bs[b] as f32 / ns[b] as f32;
        } else {
            mean_rgb[b * 3] = -1.0;
            mean_rgb[b * 3 + 1] = -1.0;
            mean_rgb[b * 3 + 2] = -1.0;
        }
    }
    Bev { height, obst_height, cls, any_obstacle, mean_rgb }
}

/// Integrate one timestamp's six cameras into a grid and finalize it.
pub fn frame_occupancy(
    preds: &[ImagePred],
    images: &[CamImage],
    poses: &PosePred,
    t: usize,
    cam: &crate::geometry::CameraModel,
    grid: &mut VoxelGrid,
) {
    grid.clear();
    for c in 0..N_CAMS {
        let pose = crate::geometry::cam_pose(poses, t, c);
        grid.integrate(&preds[t * N_CAMS + c], &images[t * N_CAMS + c].rgb, &pose, cam, c, 2, 3);
    }
    grid.finalize(3);
}

/// Ground-truth occupancy from rendered depth maps (for the IoU metric), using the
/// same ray-marching principle with GT poses (one pose per camera for frame t).
pub fn frame_occupancy_gt(
    images: &[CamImage],
    gt_poses: &[Se3],
    t: usize,
    cam: &crate::geometry::CameraModel,
    grid: &mut VoxelGrid,
) {
    grid.clear();
    for c in 0..N_CAMS {
        let mut fake_pred_depth = images[t * N_CAMS + c].depth.clone();
        for d in fake_pred_depth.iter_mut() {
            if *d >= SKY_DEPTH {
                *d = SKY_DEPTH;
            }
        }
        let pred = ImagePred {
            depth: fake_pred_depth,
            conf: vec![1.0; IMG_W * IMG_H],
            sem: images[t * N_CAMS + c].sem.clone(),
            points: vec![0.0; IMG_W * IMG_H * 3],
        };
        grid.integrate(&pred, &images[t * N_CAMS + c].rgb, &gt_poses[c], cam, c, 2, 3);
    }
    grid.finalize(3);
}

/// Occupancy IoU + semantic accuracy over voxels occupied in either grid.
pub fn occupancy_metrics(pred: &VoxelGrid, gt: &VoxelGrid) -> (f32, f32) {
    let n = pred.occupied.len();
    let mut inter = 0usize;
    let mut union = 0usize;
    let mut sem_ok = 0usize;
    let mut sem_total = 0usize;
    for i in 0..n {
        let (p, g) = (pred.occupied[i], gt.occupied[i]);
        if p == 1 && g == 1 {
            inter += 1;
            sem_total += 1;
            if pred.sem[i] == gt.sem[i] {
                sem_ok += 1;
            }
        }
        if p == 1 || g == 1 {
            union += 1;
        }
    }
    let iou = if union == 0 { 0.0 } else { inter as f32 / union as f32 };
    let sem_acc = if sem_total == 0 { 0.0 } else { sem_ok as f32 / sem_total as f32 };
    (iou, sem_acc)
}

/// Build the per-frame BEV grid over the whole clip (for detection + visualization).
pub fn clip_bevs(
    preds: &[ImagePred],
    images: &[CamImage],
    poses: &PosePred,
    cam: &crate::geometry::CameraModel,
) -> Vec<Bev> {
    let mut grid = VoxelGrid::new();
    let mut bevs = Vec::with_capacity(N_FRAMES);
    for t in 0..N_FRAMES {
        frame_occupancy(preds, images, poses, t, cam, &mut grid);
        bevs.push(bev_summary(&grid));
    }
    bevs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grid_dimensions() {
        assert_eq!(NX, 240);
        assert_eq!(NY, 120);
        assert_eq!(NZ, 16);
        let g = VoxelGrid::new();
        assert_eq!(g.hits.len(), NX * NY * NZ);
    }
}
