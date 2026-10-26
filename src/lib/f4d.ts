/**
 * Foundation4D WASM glue — loads /f4d.wasm (compiled from the Rust crate with
 * simd128) and exposes a typed API over the flat C ABI.
 */

/**
 * Stats layout (Float32Array of 48 entries):
 * [0] stages done, [1] total stages, [2] seed, [3] n windows,
 * [4] depth AbsRel, [5] occ IoU, [6] occ sem acc, [7] n detections,
 * [8] n tracks, [9] vehicles, [10] persons, [11] n gaussians,
 * [12] backbone MMAC, [13] backbone ms, [14] GFLOP, [15] GFLOP/s,
 * [16] RMSE raw, [17] RMSE stitched, [18] RMSE PGO,
 * [19] render ms, [20] tile entries, [21] blends (M), [22..31] stage ms,
 * [32] total ms, [33] frames, [34] cams, [35] img w, [36] img h,
 * [37] d, [38] layers, [39] heads, [40] tokens/img, [41] pooled tokens,
 * [42] camera group tokens, [43] temporal group tokens, [44] fps, [45] occupied voxels,
 * [46] depth AbsRel far (>30 m)
 */
export type F4DStats = Float32Array;

const IMG_W = 192;
const IMG_H = 144;
const N_CAMS = 6;
const N_FRAMES = 32;
const GW = IMG_W / 8; // 24
const GH = IMG_H / 8; // 18
const BEV_NX = 240;
const BEV_NY = 120;
const DT = 0.4;

export const F4D = { IMG_W, IMG_H, N_CAMS, N_FRAMES, GW, GH, BEV_NX, BEV_NY, DT };

export class Foundation4D {
  private exports: any;
  private memory!: WebAssembly.Memory;
  private mem(): ArrayBuffer {
    return this.memory.buffer;
  }
  private u8(ptr: number, len: number): Uint8Array {
    return new Uint8Array(this.mem(), ptr, len);
  }
  private f32(ptr: number, len: number): Float32Array {
    return new Float32Array(this.mem(), ptr, len);
  }

  static async load(): Promise<Foundation4D> {
    const res = await fetch('/f4d.wasm');
    const buf = await res.arrayBuffer();
    const { instance } = await WebAssembly.instantiate(buf, {});
    const f = new Foundation4D();
    f.exports = instance.exports;
    f.memory = instance.exports.memory as WebAssembly.Memory;
    return f;
  }

  get version(): number {
    return this.exports.f4d_version();
  }

  init(seed: number): void {
    this.exports.f4d_init(seed);
  }

  get totalStages(): number {
    return this.exports.f4d_total_stages();
  }

  /** Advance one stage; returns the completed stage index or -1 when done. */
  step(): number {
    return this.exports.f4d_step();
  }

  /** JS-measured stage timing (wasm32 has no clock). */
  setStat(idx: number, v: number): void {
    this.exports.f4d_set_stat(idx, v);
  }

  noteRender(ms: number): void {
    this.exports.f4d_note_render(ms);
  }

  stats(): F4DStats {
    const ptr = this.exports.f4d_stats_ptr();
    const len = this.exports.f4d_stats_len();
    return this.f32(ptr, len) as F4DStats;
  }

  /** RGB image of camera (t, c) — length IMG_W*IMG_H*3. */
  image(t: number, c: number): Uint8Array {
    const ptr = this.exports.f4d_image_ptr(t, c);
    if (ptr === 0) return new Uint8Array(0);
    return this.u8(ptr, IMG_W * IMG_H * 3);
  }

  gtDepth(t: number, c: number): Float32Array {
    const ptr = this.exports.f4d_gt_depth_ptr(t, c);
    if (ptr === 0) return new Float32Array(0);
    return this.f32(ptr, IMG_W * IMG_H);
  }

  gtSem(t: number, c: number): Uint8Array {
    const ptr = this.exports.f4d_gt_sem_ptr(t, c);
    if (ptr === 0) return new Uint8Array(0);
    return this.u8(ptr, IMG_W * IMG_H);
  }

  predDepth(t: number, c: number): Float32Array {
    const ptr = this.exports.f4d_pred_depth_ptr(t, c);
    if (ptr === 0) return new Float32Array(0);
    return this.f32(ptr, IMG_W * IMG_H);
  }

  predConf(t: number, c: number): Float32Array {
    const ptr = this.exports.f4d_pred_conf_ptr(t, c);
    if (ptr === 0) return new Float32Array(0);
    return this.f32(ptr, IMG_W * IMG_H);
  }

  predSem(t: number, c: number): Uint8Array {
    const ptr = this.exports.f4d_pred_sem_ptr(t, c);
    if (ptr === 0) return new Uint8Array(0);
    return this.u8(ptr, IMG_W * IMG_H);
  }

  /** Backbone contextual features (middle window, middle frame, 6 cams). */
  features(): Uint8Array {
    const ptr = this.exports.f4d_features_ptr();
    if (ptr === 0) return new Uint8Array(0);
    return this.u8(ptr, N_CAMS * GW * GH * 3);
  }

  headDepth(): Float32Array {
    const ptr = this.exports.f4d_head_depth_ptr();
    if (ptr === 0) return new Float32Array(0);
    return this.f32(ptr, N_CAMS * GW * GH * 2);
  }

  headSem(): Float32Array {
    const ptr = this.exports.f4d_head_sem_ptr();
    if (ptr === 0) return new Float32Array(0);
    return this.f32(ptr, N_CAMS * GW * GH * 8);
  }

  bevClass(t: number): Uint8Array {
    const ptr = this.exports.f4d_bev_class_ptr(t);
    if (ptr === 0) return new Uint8Array(0);
    return this.u8(ptr, BEV_NX * BEV_NY);
  }

  bevHeight(t: number): Float32Array {
    const ptr = this.exports.f4d_bev_height_ptr(t);
    if (ptr === 0) return new Float32Array(0);
    return this.f32(ptr, BEV_NX * BEV_NY);
  }

  gaussianCount(): number {
    return this.exports.f4d_gaussian_count();
  }

  // ---- Gaussian SoA (WebGPU splat renderer path). ----
  gaussPos(): Float32Array {
    const n = this.gaussianCount();
    const ptr = this.exports.f4d_gauss_pos_ptr();
    if (ptr === 0 || n === 0) return new Float32Array(0);
    return this.f32(ptr, n * 3);
  }

  gaussQuat(): Float32Array {
    const n = this.gaussianCount();
    const ptr = this.exports.f4d_gauss_quat_ptr();
    if (ptr === 0 || n === 0) return new Float32Array(0);
    return this.f32(ptr, n * 4);
  }

  gaussScale(): Float32Array {
    const n = this.gaussianCount();
    const ptr = this.exports.f4d_gauss_scale_ptr();
    if (ptr === 0 || n === 0) return new Float32Array(0);
    return this.f32(ptr, n * 3);
  }

  gaussColor(): Uint8Array {
    const n = this.gaussianCount();
    const ptr = this.exports.f4d_gauss_color_ptr();
    if (ptr === 0 || n === 0) return new Uint8Array(0);
    return this.u8(ptr, n * 3);
  }

  gaussOpacity(): Float32Array {
    const n = this.gaussianCount();
    const ptr = this.exports.f4d_gauss_opacity_ptr();
    if (ptr === 0 || n === 0) return new Float32Array(0);
    return this.f32(ptr, n);
  }

  gaussActor(): Int32Array {
    const n = this.gaussianCount();
    const ptr = this.exports.f4d_gauss_actor_ptr();
    if (ptr === 0 || n === 0) return new Int32Array(0);
    return new Int32Array(this.mem(), ptr, n);
  }

  /** Sky cubemap, 6 faces x 16x16 RGB. */
  sky(): Uint8Array {
    const ptr = this.exports.f4d_sky_ptr();
    if (ptr === 0) return new Uint8Array(0);
    return this.u8(ptr, 6 * 16 * 16 * 3);
  }

  /** Rig transforms: 6 x (quat wxyz + t), 42 floats. */
  rig(): Float32Array {
    const ptr = this.exports.f4d_rig_ptr();
    if (ptr === 0) return new Float32Array(0);
    return this.f32(ptr, 6 * 7);
  }

  /** Serialized tracks: [n, per track: id, class, dl, dw, dh, n_states, states*6]. */
  tracks(): Float32Array {
    const len = this.exports.f4d_tracks_len();
    if (len === 0) return new Float32Array(0);
    const ptr = this.exports.f4d_tracks_ptr();
    return this.f32(ptr, len);
  }

  /** Detections at frame t: [n, n * 11 (cx, cy, cz, l, w, h, yaw, class, score, vx, vy)]. */
  detections(t: number): Float32Array {
    const len = this.exports.f4d_detections_len(t);
    if (len === 0) return new Float32Array(0);
    const ptr = this.exports.f4d_detections_ptr(t);
    return this.f32(ptr, len);
  }

  /** 4 x N_FRAMES x 7 (quat wxyz + t): raw, stitched, PGO, GT. */
  trajectories(): Float32Array {
    const ptr = this.exports.f4d_traj_ptr();
    return this.f32(ptr, 4 * N_FRAMES * 7);
  }

  setRenderSize(w: number, h: number): void {
    this.exports.f4d_set_render_size(w, h);
  }

  render(
    eye: [number, number, number],
    fwd: [number, number, number],
    fov: number,
    tau: number,
    overlays: number,
  ): { ptr: number; w: number; h: number } {
    const r = this.exports.f4d_render(eye[0], eye[1], eye[2], fwd[0], fwd[1], fwd[2], fov, tau, overlays);
    if (r !== 0) {
      return { ptr: this.exports.f4d_render_ptr(), w: this.exports.f4d_render_w(), h: this.exports.f4d_render_h() };
    }
    return { ptr: 0, w: 0, h: 0 };
  }

  renderPixels(): Uint8Array {
    const w = this.exports.f4d_render_w();
    const h = this.exports.f4d_render_h();
    const ptr = this.exports.f4d_render_ptr();
    if (ptr === 0) return new Uint8Array(0);
    return this.u8(ptr, w * h * 4);
  }

  renderDims(): { w: number; h: number } {
    return { w: this.exports.f4d_render_w(), h: this.exports.f4d_render_h() };
  }
}
