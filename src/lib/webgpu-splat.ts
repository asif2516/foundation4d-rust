/// <reference types="@webgpu/types" />
/**
 * WebGPU Gaussian-splat renderer — the GPU path of the paper's scene rendering
 * (Sec 3.7). Ports the Rust software rasterizer's EWA splatting math
 * (Sigma' = J W Sigma W^T J^T, projected-size clamp, back-to-front alpha
 * compositing with the sky cubemap behind) to a 4-pass GPU pipeline:
 *
 *   1. project   (compute) — transform + EWA covariance + key per Gaussian
 *   2. histogram (compute) — 65536-bin atomic histogram of depth keys
 *   3. prefix    (compute) — single-workgroup exclusive scan over the bins
 *   4. scatter   (compute) — counting-sort into a back-to-front instance order
 *
 * then renders sky -> sorted instanced quads (indirect draw, no readback) ->
 * depth-ordered overlay lines. All GPU state is persistent; per-frame work is
 * one uniform write, four small dispatches and one indirect draw.
 */

const SHADER = /* wgsl */`

struct View {
  eye: vec4f,      // xyz
  fwd: vec4f,      // xyz (normalized)
  right: vec4f,    // xyz
  down: vec4f,     // xyz
  proj: vec4f,     // fx, fy, cx, cy (pixels)
  params: vec4f,   // w, h, far, nTracks
};

struct Record {
  u: f32, v: f32, ia: f32, ib: f32,
  ic: f32, ru: f32, rv: f32, alpha: f32,
  color: u32, key: u32,
};

const MAX_SIGMA2: f32 = 6.25; // projected-size clamp (2.5 px sigma), as in the CPU path
const INVALID: u32 = 0xFFFFFFFFu;

// ---------------------------------------------------------------------------
// Pass 1: project all Gaussians (EWA) and quantize a back-to-front sort key.
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> view: View;
@group(0) @binding(1) var<storage, read> pos: array<f32>;
@group(0) @binding(2) var<storage, read> quat: array<f32>;
@group(0) @binding(3) var<storage, read> scale: array<f32>;
@group(0) @binding(4) var<storage, read> color: array<u32>;
@group(0) @binding(5) var<storage, read> actorId: array<i32>;
@group(0) @binding(6) var<storage, read> actors: array<vec4f>; // (cx, cy, cz, yaw)
@group(0) @binding(7) var<storage, read_write> records: array<Record>;

@compute @workgroup_size(128)
fn projectMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&records)) { return; }
  var rec = Record(0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0u, INVALID);
  let a = actorId[i];
  var p = vec3f(pos[3u * i], pos[3u * i + 1u], pos[3u * i + 2u]);
  if (a >= 0) {
    // Actor-local frame -> world at render time (tracked cuboid transform).
    let t = actors[u32(a)];
    let c = cos(t.w); let s = sin(t.w);
    let lx = p.x; let ly = p.y;
    p = vec3f(t.x + c * lx - s * ly, t.y + s * lx + c * ly, t.z + p.z);
  }
  let eye = view.eye.xyz;
  let fwd = view.fwd.xyz;
  let right = view.right.xyz;
  let down = view.down.xyz;
  let d = p - eye;
  let z = dot(d, fwd);
  let far = view.params.z;
  let w = view.params.x;
  let h = view.params.y;
  if (z >= 1.2 && z <= far) {
    let x = dot(d, right);
    let y = dot(d, down);
    let fx = view.proj.x; let fy = view.proj.y;
    let cx = view.proj.z; let cy = view.proj.w;
    let invz = 1.0 / z;
    let u = cx + fx * x * invz;
    let v = cy + fy * y * invz;
    // R from quaternion (wxyz).
    let q = vec4f(quat[4u * i], quat[4u * i + 1u], quat[4u * i + 2u], quat[4u * i + 3u]);
    let r00 = 1.0 - 2.0 * (q.y * q.y + q.z * q.z);
    let r01 = 2.0 * (q.x * q.y - q.w * q.z);
    let r02 = 2.0 * (q.x * q.z + q.w * q.y);
    let r10 = 2.0 * (q.x * q.y + q.w * q.z);
    let r11 = 1.0 - 2.0 * (q.x * q.x + q.z * q.z);
    let r12 = 2.0 * (q.y * q.z - q.w * q.x);
    let r20 = 2.0 * (q.x * q.z - q.w * q.y);
    let r21 = 2.0 * (q.y * q.z + q.w * q.x);
    let r22 = 1.0 - 2.0 * (q.x * q.x + q.y * q.y);
    // M = W * R (W rows: right / down / fwd).
    let m00 = right.x * r00 + right.y * r10 + right.z * r20;
    let m01 = right.x * r01 + right.y * r11 + right.z * r21;
    let m02 = right.x * r02 + right.y * r12 + right.z * r22;
    let m10 = down.x * r00 + down.y * r10 + down.z * r20;
    let m11 = down.x * r01 + down.y * r11 + down.z * r21;
    let m12 = down.x * r02 + down.y * r12 + down.z * r22;
    let m20 = fwd.x * r00 + fwd.y * r10 + fwd.z * r20;
    let m21 = fwd.x * r01 + fwd.y * r11 + fwd.z * r21;
    let m22 = fwd.x * r02 + fwd.y * r12 + fwd.z * r22;
    let s0 = scale[3u * i]; let s1 = scale[3u * i + 1u]; let s2 = scale[3u * i + 2u];
    let a0 = s0 * s0; let a1 = s1 * s1; let a2 = s2 * s2;
    let sc00 = m00 * m00 * a0 + m01 * m01 * a1 + m02 * m02 * a2;
    let sc11 = m10 * m10 * a0 + m11 * m11 * a1 + m12 * m12 * a2;
    let sc22 = m20 * m20 * a0 + m21 * m21 * a1 + m22 * m22 * a2;
    let sc01 = m00 * m10 * a0 + m01 * m11 * a1 + m02 * m12 * a2;
    let sc02 = m00 * m20 * a0 + m01 * m21 * a1 + m02 * m22 * a2;
    let sc12 = m10 * m20 * a0 + m11 * m21 * a1 + m12 * m22 * a2;
    let j00 = fx * invz;
    let j02 = -fx * x * invz * invz;
    let j11 = fy * invz;
    let j12 = -fy * y * invz * invz;
    var sa = j00 * j00 * sc00 + j02 * j02 * sc22 + 2.0 * j00 * j02 * sc02;
    var sb = j00 * j11 * sc01 + j00 * j12 * sc02 + j02 * j11 * sc12 + j02 * j12 * sc22;
    var sc = j11 * j11 * sc11 + j12 * j12 * sc22 + 2.0 * j11 * j12 * sc12;
    let smax = max(sa, sc);
    if (smax > MAX_SIGMA2) {
      let f = MAX_SIGMA2 / smax;
      sa *= f; sb *= f; sc *= f;
    }
    let det = sa * sc - sb * sb;
    let ru = 3.0 * sqrt(sa);
    let rv = 3.0 * sqrt(sc);
    let margin = 32.0;
    if (det > 1e-12
        && u + ru >= -margin && u - ru <= w + margin
        && v + rv >= -margin && v - rv <= h + margin) {
      let inv = 1.0 / det;
      rec.u = u;
      rec.v = v;
      rec.ia = sc * inv;
      rec.ib = -sb * inv;
      rec.ic = sa * inv;
      rec.ru = ru;
      rec.rv = rv;
      rec.alpha = unpack4x8unorm(color[i]).w;
      rec.color = color[i];
      // Back-to-front: ascending key order = far first.
      let kf = clamp(z / far, 0.0, 1.0) * 65535.0;
      rec.key = u32(65535.0 - kf);
    }
  }
  records[i] = rec;
}

// ---------------------------------------------------------------------------
// Pass 2: depth-key histogram (valid Gaussians only).
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read_write> bins: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> recordsH: array<Record>;

@compute @workgroup_size(128)
fn histogramMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&recordsH)) { return; }
  let k = recordsH[i].key;
  if (k == INVALID) { return; }
  atomicAdd(&bins[k], 1u);
}

@group(0) @binding(0) var<storage, read_write> binsClear: array<atomic<u32>>;
@compute @workgroup_size(128)
fn clearMain(@builtin(global_invocation_id) gid: vec3u) {
  if (gid.x < 65536u) { atomicStore(&binsClear[gid.x], 0u); }
}

// ---------------------------------------------------------------------------
// Pass 3: exclusive scan over the 65536 bins (one workgroup, 256 threads).
// Writes per-bin starting offsets and the indirect draw parameters.
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read_write> binsP: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read_write> offsets: array<u32>;
@group(0) @binding(2) var<storage, read_write> indirect: array<u32>;
var<workgroup> sums: array<u32, 256>;

@compute @workgroup_size(256)
fn prefixMain(@builtin(local_invocation_id) lid: vec3u) {
  let tid = lid.x;
  var run = 0u;
  for (var j = 0u; j < 256u; j++) {
    let idx = tid * 256u + j;
    offsets[idx] = run;
    run += atomicLoad(&binsP[idx]);
  }
  sums[tid] = run;
  workgroupBarrier();
  var stride = 1u;
  while (stride < 256u) {
    var v = 0u;
    if (tid >= stride) { v = sums[tid - stride]; }
    workgroupBarrier();
    sums[tid] = sums[tid] + v;
    workgroupBarrier();
    stride *= 2u;
  }
  let excl = sums[tid] - run;
  for (var j = 0u; j < 256u; j++) {
    offsets[tid * 256u + j] += excl;
  }
  if (tid == 255u) {
    indirect[0] = 6u;          // vertices per Gaussian (2 triangles)
    indirect[1] = sums[255];   // instance count = visible Gaussians
    indirect[2] = 0u;
    indirect[3] = 0u;
    indirect[4] = 0u;
  }
}

// ---------------------------------------------------------------------------
// Pass 4: counting-sort scatter into the back-to-front instance order.
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read_write> cursor: array<atomic<u32>>;
@group(0) @binding(1) var<storage, read> recordsS: array<Record>;
@group(0) @binding(2) var<storage, read_write> order: array<u32>;

@compute @workgroup_size(128)
fn scatterMain(@builtin(global_invocation_id) gid: vec3u) {
  let i = gid.x;
  if (i >= arrayLength(&recordsS)) { return; }
  let k = recordsS[i].key;
  if (k == INVALID) { return; }
  let at = atomicAdd(&cursor[k], 1u);
  order[at] = i;
}

// ---------------------------------------------------------------------------
// Sky pass: fullscreen triangle sampling the reconstructed sky cubemap.
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<uniform> viewSky: View;
@group(0) @binding(1) var<storage, read> sky: array<u32>; // 6*16*16 packed rgb

fn skyFaceTexel(ray: vec3f) -> vec3f {
  let ax = abs(ray.x); let ay = abs(ray.y); let az = abs(ray.z);
  var face = 0u; var uu = 0.0; var vv = 0.0;
  if (ax >= ay && ax >= az) {
    if (ray.x > 0.0) { face = 0u; uu = ray.y / ax; vv = ray.z / ax; }
    else { face = 1u; uu = -ray.y / ax; vv = ray.z / ax; }
  } else if (ay >= ax && ay >= az) {
    if (ray.y > 0.0) { face = 2u; uu = -ray.x / ay; vv = ray.z / ay; }
    else { face = 3u; uu = ray.x / ay; vv = ray.z / ay; }
  } else if (ray.z > 0.0) {
    face = 4u; uu = ray.x / az; vv = ray.y / az;
  } else {
    face = 5u; uu = ray.x / az; vv = -ray.y / az;
  }
  let tu = u32(round((uu * 0.5 + 0.5) * 15.0));
  let tv = u32(round((vv * 0.5 + 0.5) * 15.0));
  let idx = face * 256u + tv * 16u + tu;
  return unpack4x8unorm(sky[idx]).rgb;
}

struct SkyOut { @builtin(position) pos: vec4f, @location(0) ndc: vec2f };

@vertex
fn vsSky(@builtin(vertex_index) vi: u32) -> SkyOut {
  var p = array<vec2f, 3>(vec2f(-1.0, -1.0), vec2f(3.0, -1.0), vec2f(-1.0, 3.0));
  var o: SkyOut;
  o.pos = vec4f(p[vi], 0.0, 1.0);
  o.ndc = p[vi];
  return o;
}

@fragment
fn fsSky(in: SkyOut) -> @location(0) vec4f {
  let w = viewSky.params.x; let h = viewSky.params.y;
  let fx = viewSky.proj.x; let fy = viewSky.proj.y;
  let cx = viewSky.proj.z; let cy = viewSky.proj.w;
  let px = (in.ndc.x * 0.5 + 0.5) * w - 0.5;
  let py = (0.5 - in.ndc.y * 0.5) * h - 0.5;
  let sx = (px - cx) / fx;
  let sy = (py - cy) / fy;
  let dir = normalize(viewSky.fwd.xyz + viewSky.right.xyz * sx + viewSky.down.xyz * sy);
  return vec4f(skyFaceTexel(dir), 1.0);
}

// ---------------------------------------------------------------------------
// Splat pass: instanced EWA quads in ascending-key (back-to-front) order.
// ---------------------------------------------------------------------------
@group(0) @binding(0) var<storage, read> recordsR: array<Record>;
@group(0) @binding(1) var<storage, read> orderR: array<u32>;
@group(0) @binding(2) var<uniform> viewR: View;

struct SplatOut {
  @builtin(position) pos: vec4f,
  @location(0) off: vec2f,
  @location(1) cov: vec4f,   // ia, ib, ic, alpha
  @location(2) color: vec4f,
};

@vertex
fn vsSplat(@builtin(vertex_index) vi: u32, @builtin(instance_index) ii: u32) -> SplatOut {
  let rec = recordsR[orderR[ii]];
  var corner = array<vec2f, 6>(
    vec2f(-1.0, -1.0), vec2f(1.0, -1.0), vec2f(1.0, 1.0),
    vec2f(-1.0, -1.0), vec2f(1.0, 1.0), vec2f(-1.0, 1.0));
  let c = corner[vi];
  let u = rec.u + c.x * rec.ru;
  let v = rec.v + c.y * rec.rv;
  let w = viewR.params.x; let h = viewR.params.y;
  var o: SplatOut;
  o.pos = vec4f(u / w * 2.0 - 1.0, 1.0 - v / h * 2.0, 0.0, 1.0);
  o.off = vec2f(c.x * rec.ru, c.y * rec.rv);
  o.cov = vec4f(rec.ia, rec.ib, rec.ic, rec.alpha);
  o.color = unpack4x8unorm(rec.color);
  return o;
}

@fragment
fn fsSplat(in: SplatOut) -> @location(0) vec4f {
  let du = in.off.x; let dv = in.off.y;
  let q = in.cov.x * du * du + 2.0 * in.cov.y * du * dv + in.cov.z * dv * dv;
  if (q > 9.0) { discard; }
  let alpha = in.cov.w * exp(-0.5 * q);
  if (alpha < 0.004) { discard; }
  return vec4f(in.color.rgb, alpha);
}

// ---------------------------------------------------------------------------
// Overlay lines (cuboids / trajectory / frustums), pre-projected on the CPU.
// ---------------------------------------------------------------------------
struct LineIn {
  @location(0) pos: vec2f, // pixel coords
  @location(1) col: vec3f,
};

struct LineOut {
  @builtin(position) pos: vec4f,
  @location(0) col: vec3f,
};

@group(0) @binding(0) var<uniform> viewL: View;

@vertex
fn vsLine(in: LineIn) -> LineOut {
  let w = viewL.params.x; let h = viewL.params.y;
  var o: LineOut;
  o.pos = vec4f(in.pos.x / w * 2.0 - 1.0, 1.0 - in.pos.y / h * 2.0, 0.0, 1.0);
  o.col = in.col;
  return o;
}

@fragment
fn fsLine(in: LineOut) -> @location(0) vec4f {
  return vec4f(in.col, 1.0);
}
`;

// ---------------------------------------------------------------------------

export interface TrackParsed {
  id: number;
  cls: number;
  dims: [number, number, number];
  states: Float32Array; // 6 per state: t, x, y, vx, vy, yaw
}

export interface SceneData {
  n: number;
  pos: Float32Array;
  quat: Float32Array;
  scale: Float32Array;
  colorPacked: Uint32Array;
  skyPacked: Uint32Array;
}

export interface FrameParams {
  eye: [number, number, number];
  fwd: [number, number, number];
  fovY: number;
  tau: number;
  overlays: number;
  tracks: TrackParsed[];
  traj: Float32Array; // assembled translations: NF * (x, y, z) — actually NF*7 pose rows
  rig: Float32Array; // 6 * (quat wxyz + t)
}

const BINS = 65536;
const MAX_ACTORS = 64;
const REC_STRIDE = 10; // f32/u32 per record

function trackStateAt(tr: TrackParsed, tau: number): [number, number, number, number] {
  const s = tr.states;
  const n = s.length / 6;
  if (n === 0) return [0, 0, 0.75, 0];
  const dt = 0.4;
  if (tau <= s[0] * dt) {
    const st = 0;
    const d = tau - s[st * 6] * dt;
    return [s[st * 6 + 1] + s[st * 6 + 3] * d, s[st * 6 + 2] + s[st * 6 + 4] * d, 0.75, s[st * 6 + 5]];
  }
  if (tau >= s[(n - 1) * 6] * dt) {
    const st = n - 1;
    const d = tau - s[st * 6] * dt;
    return [s[st * 6 + 1] + s[st * 6 + 3] * d, s[st * 6 + 2] + s[st * 6 + 4] * d, 0.75, s[st * 6 + 5]];
  }
  for (let i = 1; i < n; i++) {
    const t1 = s[i * 6] * dt;
    if (tau <= t1) {
      const t0 = s[(i - 1) * 6] * dt;
      const a = (tau - t0) / Math.max(t1 - t0, 1e-6);
      return [
        s[(i - 1) * 6 + 1] + (s[i * 6 + 1] - s[(i - 1) * 6 + 1]) * a,
        s[(i - 1) * 6 + 2] + (s[i * 6 + 2] - s[(i - 1) * 6 + 2]) * a,
        0.75,
        s[(i - 1) * 6 + 5] + (s[i * 6 + 5] - s[(i - 1) * 6 + 5]) * a,
      ];
    }
  }
  return [s[1], s[2], 0.75, s[5]];
}

export class WebGpuSplatRenderer {
  private device: GPUDevice;
  private ctx: GPUCanvasContext;
  private format: GPUTextureFormat;
  private module: GPUShaderModule;
  // Pipelines.
  private pipeProject!: GPUComputePipeline;
  private pipeHistogram!: GPUComputePipeline;
  private pipeClear!: GPUComputePipeline;
  private pipePrefix!: GPUComputePipeline;
  private pipeScatter!: GPUComputePipeline;
  private pipeSky!: GPURenderPipeline;
  private pipeSplat!: GPURenderPipeline;
  private pipeLine!: GPURenderPipeline;
  // Bind groups.
  private bgProject!: GPUBindGroup;
  private bgHistogram!: GPUBindGroup;
  private bgClear!: GPUBindGroup;
  private bgPrefix!: GPUBindGroup;
  private bgScatter!: GPUBindGroup;
  private bgSky!: GPUBindGroup;
  private bgSplat!: GPUBindGroup;
  private bgLine!: GPUBindGroup;
  // Buffers.
  private bufPos!: GPUBuffer;
  private bufQuat!: GPUBuffer;
  private bufScale!: GPUBuffer;
  private bufColor!: GPUBuffer;
  private bufActor!: GPUBuffer;
  private bufActors!: GPUBuffer; // per-frame actor transforms
  private bufRecords!: GPUBuffer;
  private bufBins!: GPUBuffer;
  private bufOffsets!: GPUBuffer;
  private bufCursor!: GPUBuffer;
  private bufOrder!: GPUBuffer;
  private bufIndirect!: GPUBuffer;
  private bufSky!: GPUBuffer;
  private bufUniform!: GPUBuffer;
  private bufLines!: GPUBuffer;
  private nGauss = 0;
  private lineVerts = new Float32Array(6000 * 5);
  private ready = false;

  constructor(device: GPUDevice, ctx: GPUCanvasContext, format: GPUTextureFormat) {
    this.device = device;
    this.ctx = ctx;
    this.format = format;
    this.module = device.createShaderModule({ code: SHADER, label: 'f4d-splat' });
  }

  static async create(canvas: HTMLCanvasElement): Promise<WebGpuSplatRenderer | null> {
    if (typeof navigator === 'undefined' || !(navigator as any).gpu) return null;
    try {
      const adapter = await (navigator as any).gpu.requestAdapter({ powerPreference: 'high-performance' });
      if (!adapter) return null;
      const device = await adapter.requestDevice();
      const ctx = canvas.getContext('webgpu') as GPUCanvasContext | null;
      if (!ctx) return null;
      const format = (navigator as any).gpu.getPreferredCanvasFormat();
      ctx.configure({ device, format, alphaMode: 'opaque' });
      device.lost.then(() => { /* context lost — caller falls back */ });
      return new WebGpuSplatRenderer(device, ctx, format);
    } catch {
      return null;
    }
  }

  /** Upload the Gaussian cloud + sky (once per pipeline run). */
  uploadScene(data: SceneData): void {
    const dev = this.device;
    this.nGauss = data.n;
    const usage: GPUBufferUsageFlags = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.bufPos = dev.createBuffer({ size: align(data.pos.byteLength), usage });
    this.bufQuat = dev.createBuffer({ size: align(data.quat.byteLength), usage });
    this.bufScale = dev.createBuffer({ size: align(data.scale.byteLength), usage });
    this.bufColor = dev.createBuffer({ size: align(data.colorPacked.byteLength), usage });
    this.bufActor = dev.createBuffer({ size: align(Math.max(4, data.n * 4)), usage });
    this.bufSky = dev.createBuffer({ size: align(data.skyPacked.byteLength), usage });
    dev.queue.writeBuffer(this.bufPos, 0, data.pos as any);
    dev.queue.writeBuffer(this.bufQuat, 0, data.quat as any);
    dev.queue.writeBuffer(this.bufScale, 0, data.scale as any);
    dev.queue.writeBuffer(this.bufColor, 0, data.colorPacked as any);
    dev.queue.writeBuffer(this.bufSky, 0, data.skyPacked as any);

    this.bufRecords = dev.createBuffer({ size: Math.max(16, data.n * REC_STRIDE * 4), usage: GPUBufferUsage.STORAGE });
    this.bufBins = dev.createBuffer({ size: BINS * 4, usage: GPUBufferUsage.STORAGE });
    this.bufOffsets = dev.createBuffer({ size: BINS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
    this.bufCursor = dev.createBuffer({ size: BINS * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.bufOrder = dev.createBuffer({ size: align(Math.max(4, data.n * 4)), usage: GPUBufferUsage.STORAGE });
    this.bufIndirect = dev.createBuffer({ size: 32, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.INDIRECT | GPUBufferUsage.COPY_DST });
    this.bufActors = dev.createBuffer({ size: MAX_ACTORS * 16, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST });
    this.bufUniform = dev.createBuffer({ size: 96, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    this.bufLines = dev.createBuffer({ size: this.lineVerts.byteLength, usage: GPUBufferUsage.VERTEX | GPUBufferUsage.COPY_DST });
    // Zero the indirect draw so nothing renders before the first frame.
    dev.queue.writeBuffer(this.bufIndirect, 0, new Uint32Array([0, 0, 0, 0, 0, 0, 0, 0]));

    this.buildPipelines();
    this.ready = true;
  }

  /** Gaussian actor ids (uploaded separately from SceneData). */
  uploadActorIds(ids: Int32Array): void {
    if (ids.length) this.device.queue.writeBuffer(this.bufActor, 0, ids as any);
  }

  private buildPipelines(): void {
    const dev = this.device;
    const mod = this.module;

    this.pipeProject = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'projectMain' } });
    this.pipeHistogram = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'histogramMain' } });
    this.pipeClear = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'clearMain' } });
    this.pipePrefix = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'prefixMain' } });
    this.pipeScatter = dev.createComputePipeline({ layout: 'auto', compute: { module: mod, entryPoint: 'scatterMain' } });

    const blend: GPUBlendState = {
      color: { srcFactor: 'src-alpha', dstFactor: 'one-minus-src-alpha', operation: 'add' },
      alpha: { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' },
    };
    this.pipeSky = dev.createRenderPipeline({
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vsSky' },
      fragment: { module: mod, entryPoint: 'fsSky', targets: [{ format: this.format }] },
      primitive: { topology: 'triangle-list' },
    });
    this.pipeSplat = dev.createRenderPipeline({
      layout: 'auto',
      vertex: { module: mod, entryPoint: 'vsSplat' },
      fragment: { module: mod, entryPoint: 'fsSplat', targets: [{ format: this.format, blend }] },
      primitive: { topology: 'triangle-list' },
    });
    this.pipeLine = dev.createRenderPipeline({
      layout: 'auto',
      vertex: {
        module: mod, entryPoint: 'vsLine',
        buffers: [{
          arrayStride: 20,
          attributes: [
            { shaderLocation: 0, offset: 0, format: 'float32x2' },
            { shaderLocation: 1, offset: 8, format: 'float32x3' },
          ],
        }],
      },
      fragment: { module: mod, entryPoint: 'fsLine', targets: [{ format: this.format }] },
      primitive: { topology: 'line-list' },
    });

    this.bgProject = dev.createBindGroup({
      layout: this.pipeProject.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufUniform } },
        { binding: 1, resource: { buffer: this.bufPos } },
        { binding: 2, resource: { buffer: this.bufQuat } },
        { binding: 3, resource: { buffer: this.bufScale } },
        { binding: 4, resource: { buffer: this.bufColor } },
        { binding: 5, resource: { buffer: this.bufActor } },
        { binding: 6, resource: { buffer: this.bufActors } },
        { binding: 7, resource: { buffer: this.bufRecords } },
      ],
    });
    this.bgClear = dev.createBindGroup({
      layout: this.pipeClear.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.bufBins } }],
    });
    this.bgHistogram = dev.createBindGroup({
      layout: this.pipeHistogram.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufBins } },
        { binding: 1, resource: { buffer: this.bufRecords } },
      ],
    });
    this.bgPrefix = dev.createBindGroup({
      layout: this.pipePrefix.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufBins } },
        { binding: 1, resource: { buffer: this.bufOffsets } },
        { binding: 2, resource: { buffer: this.bufIndirect } },
      ],
    });
    this.bgScatter = dev.createBindGroup({
      layout: this.pipeScatter.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufCursor } },
        { binding: 1, resource: { buffer: this.bufRecords } },
        { binding: 2, resource: { buffer: this.bufOrder } },
      ],
    });
    this.bgSky = dev.createBindGroup({
      layout: this.pipeSky.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufUniform } },
        { binding: 1, resource: { buffer: this.bufSky } },
      ],
    });
    this.bgSplat = dev.createBindGroup({
      layout: this.pipeSplat.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: this.bufRecords } },
        { binding: 1, resource: { buffer: this.bufOrder } },
        { binding: 2, resource: { buffer: this.bufUniform } },
      ],
    });
    this.bgLine = dev.createBindGroup({
      layout: this.pipeLine.getBindGroupLayout(0),
      entries: [{ binding: 0, resource: { buffer: this.bufUniform } }],
    });
  }

  /** Render one frame. Returns approximate GPU-side drawn-instance count. */
  render(params: FrameParams, w: number, h: number): number {
    if (!this.ready || this.nGauss === 0) return 0;
    const dev = this.device;
    const [ex, ey, ez] = params.eye;
    const fwd = normalize3(params.fwd);
    const up = [0, 0, 1] as const;
    const right = normalize3(cross3(fwd, up));
    const down = cross3(right, fwd);
    const fy = (h * 0.5) / Math.tan(params.fovY * 0.5);
    const fx = fy;
    const cx = w * 0.5;
    const cy = h * 0.5;
    const far = 95.0;

    // Uniform: eye, fwd, right, down, proj, params.
    const uni = new Float32Array(24);
    uni.set([ex, ey, ez, 0], 0);
    uni.set([fwd[0], fwd[1], fwd[2], 0], 4);
    uni.set([right[0], right[1], right[2], 0], 8);
    uni.set([down[0], down[1], down[2], 0], 12);
    uni.set([fx, fy, cx, cy], 16);
    const nTracks = Math.min(params.tracks.length, MAX_ACTORS);
    uni.set([w, h, far, nTracks], 20);
    dev.queue.writeBuffer(this.bufUniform, 0, uni);

    // Actor transforms at tau.
    const actorData = new Float32Array(MAX_ACTORS * 4);
    for (let i = 0; i < nTracks; i++) {
      const tr = params.tracks[i];
      const [x, y, z, yaw] = trackStateAt(tr, params.tau);
      actorData.set([x, y, z + tr.dims[2] * 0.5, yaw], i * 4);
    }
    dev.queue.writeBuffer(this.bufActors, 0, actorData);

    // Overlay line vertices (CPU-projected).
    let lv = 0;
    const project = (p: [number, number, number]): [number, number] | null => {
      const dx = p[0] - ex, dy = p[1] - ey, dz = p[2] - ez;
      const z = dx * fwd[0] + dy * fwd[1] + dz * fwd[2];
      if (z < 0.2) return null;
      const x = dx * right[0] + dy * right[1] + dz * right[2];
      const y = dx * down[0] + dy * down[1] + dz * down[2];
      return [w * 0.5 + fy * x / z, h * 0.5 + fy * y / z];
    };
    const line = (a: [number, number, number], b: [number, number, number], col: [number, number, number]) => {
      const pa = project(a);
      const pb = project(b);
      if (!pa || !pb || lv + 10 > this.lineVerts.length) return;
      this.lineVerts[lv++] = pa[0]; this.lineVerts[lv++] = pa[1];
      this.lineVerts[lv++] = col[0]; this.lineVerts[lv++] = col[1]; this.lineVerts[lv++] = col[2];
      this.lineVerts[lv++] = pb[0]; this.lineVerts[lv++] = pb[1];
      this.lineVerts[lv++] = col[0]; this.lineVerts[lv++] = col[1]; this.lineVerts[lv++] = col[2];
    };
    if (params.overlays & 1) {
      for (const tr of params.tracks) {
        const [x, y, z, yaw] = trackStateAt(tr, params.tau);
        const col: [number, number, number] = tr.cls === 6 ? [1.0, 0.47, 0.24] : [0.31, 0.86, 0.47];
        const cy = Math.cos(yaw);
        const sy = Math.sin(yaw);
        const hd: [number, number, number] = [tr.dims[0] * 0.5, tr.dims[1] * 0.5, tr.dims[2] * 0.5];
        const pts: [number, number, number][] = [];
        for (let k = 0; k < 8; k++) {
          const sx = (k & 1) ? 1 : -1, syy = (k & 2) ? 1 : -1, sz = (k & 4) ? 1 : -1;
          const lx = sx * hd[0], ly = syy * hd[1], lz = sz * hd[2] + tr.dims[2] * 0.5;
          pts.push([x + cy * lx - sy * ly, y + sy * lx + cy * ly, z + lz]);
        }
        const edges = [[0, 1], [1, 3], [3, 2], [2, 0], [4, 5], [5, 7], [7, 6], [6, 4], [0, 4], [1, 5], [2, 6], [3, 7]];
        for (const [a, b] of edges) line(pts[a], pts[b], col);
      }
    }
    if (params.overlays & 2) {
      const col: [number, number, number] = [1.0, 0.82, 0.24];
      const nf = params.traj.length / 7;
      for (let k = 1; k < nf; k++) {
        const a: [number, number, number] = [params.traj[(k - 1) * 7 + 4], params.traj[(k - 1) * 7 + 5], params.traj[(k - 1) * 7 + 6]];
        const b: [number, number, number] = [params.traj[k * 7 + 4], params.traj[k * 7 + 5], params.traj[k * 7 + 6]];
        line(a, b, col);
      }
      // Rig marker at the nearest keyframe.
      const fi = Math.max(0, Math.min(nf - 1, Math.round(params.tau / 0.4)));
      const px = params.traj[fi * 7 + 4], py = params.traj[fi * 7 + 5], pz = params.traj[fi * 7 + 6];
      const q = [params.traj[fi * 7], params.traj[fi * 7 + 1], params.traj[fi * 7 + 2], params.traj[fi * 7 + 3]];
      const rot = quatToMat(q);
      const axes: [number, number, number][] = [[1.2, 0, 0], [0, 1.2, 0], [0, 0, 1.2]];
      const cols: [number, number, number][] = [[1, 0.31, 0.31], [0.31, 1, 0.31], [0.31, 0.31, 1]];
      for (let k = 0; k < 3; k++) {
        const wv: [number, number, number] = [
          rot[0] * axes[k][0] + rot[1] * axes[k][1] + rot[2] * axes[k][2] + px,
          rot[3] * axes[k][0] + rot[4] * axes[k][1] + rot[5] * axes[k][2] + py,
          rot[6] * axes[k][0] + rot[7] * axes[k][1] + rot[8] * axes[k][2] + pz,
        ];
        line([px, py, pz], wv, cols[k]);
      }
      for (let c = 0; c < 6; c++) {
        const cp = rigPose(params.rig, c, [px, py, pz], q);
        line([cp[0], cp[1], cp[2]], [cp[0], cp[1], cp[2] + 0.35], [0.78, 0.78, 0.86]);
      }
    }
    if (params.overlays & 4) {
      const nf = params.traj.length / 7;
      const fi = Math.max(0, Math.min(nf - 1, Math.round(params.tau / 0.4)));
      const px = params.traj[fi * 7 + 4], py = params.traj[fi * 7 + 5], pz = params.traj[fi * 7 + 6];
      const q = [params.traj[fi * 7], params.traj[fi * 7 + 1], params.traj[fi * 7 + 2], params.traj[fi * 7 + 3]];
      const col: [number, number, number] = [0.47, 0.55, 1.0];
      for (let c = 0; c < 6; c++) {
        const cp = rigPose(params.rig, c, [px, py, pz], q);
        const o: [number, number, number] = [cp[0], cp[1], cp[2]];
        const rot = cp.slice(3) as unknown as number[];
        const tip: [number, number, number] = [
          rot[0] * 0 + rot[1] * 0 + rot[2] * 0.9 + o[0],
          rot[3] * 0 + rot[4] * 0 + rot[5] * 0.9 + o[1],
          rot[6] * 0 + rot[7] * 0 + rot[8] * 0.9 + o[2],
        ];
        line(o, tip, col);
        for (const [dx, dy] of [[-0.3, -0.22], [0.3, -0.22], [0.3, 0.22], [-0.3, 0.22]]) {
          const corner: [number, number, number] = [
            rot[0] * dx + rot[1] * dy + rot[2] * 0.7 + o[0],
            rot[3] * dx + rot[4] * dy + rot[5] * 0.7 + o[1],
            rot[6] * dx + rot[7] * dy + rot[8] * 0.7 + o[2],
          ];
          line(tip, corner, col);
        }
      }
    }
    if (lv > 0) {
      dev.queue.writeBuffer(this.bufLines, 0, this.lineVerts, 0, lv);
    }

    const enc = dev.createCommandEncoder();
    const groups = Math.ceil(this.nGauss / 128);
    // Clear bins.
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.pipeClear);
      p.setBindGroup(0, this.bgClear);
      p.dispatchWorkgroups(Math.ceil(BINS / 128));
      p.end();
    }
    // Project.
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.pipeProject);
      p.setBindGroup(0, this.bgProject);
      p.dispatchWorkgroups(groups);
      p.end();
    }
    // Histogram.
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.pipeHistogram);
      p.setBindGroup(0, this.bgHistogram);
      p.dispatchWorkgroups(groups);
      p.end();
    }
    // offsets -> cursor (destructive copy target), then prefix scan.
    enc.copyBufferToBuffer(this.bufOffsets, 0, this.bufCursor, 0, BINS * 4);
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.pipePrefix);
      p.setBindGroup(0, this.bgPrefix);
      p.dispatchWorkgroups(1);
      p.end();
    }
    // Scatter.
    {
      const p = enc.beginComputePass();
      p.setPipeline(this.pipeScatter);
      p.setBindGroup(0, this.bgScatter);
      p.dispatchWorkgroups(groups);
      p.end();
    }
    // Render.
    const view = this.ctx.getCurrentTexture().createView();
    const rp = enc.beginRenderPass({
      colorAttachments: [{
        view,
        clearValue: { r: 0, g: 0, b: 0, a: 1 },
        loadOp: 'clear',
        storeOp: 'store',
      }],
    });
    rp.setPipeline(this.pipeSky);
    rp.setBindGroup(0, this.bgSky);
    rp.draw(3);
    rp.setPipeline(this.pipeSplat);
    rp.setBindGroup(0, this.bgSplat);
    rp.drawIndirect(this.bufIndirect, 0);
    if (lv > 0) {
      rp.setPipeline(this.pipeLine);
      rp.setBindGroup(0, this.bgLine);
      rp.setVertexBuffer(0, this.bufLines);
      rp.draw(lv / 5);
    }
    rp.end();
    dev.queue.submit([enc.finish()]);
    return this.nGauss;
  }

  destroy(): void {
    try { this.device.destroy(); } catch { /* ignore */ }
  }
}

// ---- small vector helpers ----

function normalize3(v: [number, number, number]): [number, number, number] {
  const n = Math.hypot(v[0], v[1], v[2]) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}

function cross3(a: readonly [number, number, number], b: readonly [number, number, number]): [number, number, number] {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function quatToMat(q: number[]): number[] {
  const [qw, qx, qy, qz] = q;
  return [
    1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qw * qz), 2 * (qx * qz + qw * qy),
    2 * (qx * qy + qw * qz), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qw * qx),
    2 * (qx * qz - qw * qy), 2 * (qy * qz + qw * qx), 1 - 2 * (qx * qx + qy * qy),
  ];
}

/** Compose rig pose (quat wxyz + t) with the ego pose (position + quat). */
function rigPose(
  rig: Float32Array, c: number, egoP: [number, number, number], egoQ: number[],
): number[] {
  const qw = rig[c * 7], qx = rig[c * 7 + 1], qy = rig[c * 7 + 2], qz = rig[c * 7 + 3];
  const tx = rig[c * 7 + 4], ty = rig[c * 7 + 5], tz = rig[c * 7 + 6];
  const a = quatToMat(egoQ);
  const b = quatToMat([qw, qx, qy, qz]);
  // R = A * B (row-major 3x3 multiply).
  const r = new Array(9);
  for (let i = 0; i < 3; i++) {
    for (let j = 0; j < 3; j++) {
      r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
    }
  }
  const t = [
    a[0] * tx + a[1] * ty + a[2] * tz + egoP[0],
    a[3] * tx + a[4] * ty + a[5] * tz + egoP[1],
    a[6] * tx + a[7] * ty + a[8] * tz + egoP[2],
  ];
  return [...t, ...r];
}

function align(bytes: number): number {
  return Math.max(16, Math.ceil(bytes / 16) * 16);
}

/** Parse the serialized track list ([n, per track: id, class, dl, dw, dh, n_states, states*6]). */
export function parseTracks(data: Float32Array): TrackParsed[] {
  const out: TrackParsed[] = [];
  if (!data || data.length === 0) return out;
  const n = data[0];
  let o = 1;
  for (let i = 0; i < n; i++) {
    const id = data[o];
    const cls = data[o + 1];
    const dims: [number, number, number] = [data[o + 2], data[o + 3], data[o + 4]];
    const ns = data[o + 5];
    o += 6;
    const states = data.slice(o, o + ns * 6);
    o += ns * 6;
    out.push({ id, cls, dims, states });
  }
  return out;
}
