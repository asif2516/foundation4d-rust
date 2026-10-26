'use client';

/**
 * 3D Reconstruction view — Gaussian-splat rendering of the reconstructed scene.
 *
 * Two render paths:
 *   * WebGPU (default when available): the EWA splatting pipeline of
 *     webgpu-splat.ts — GPU projection, GPU counting sort, instanced quads,
 *     sky cubemap and overlay lines — targeting 60 fps at full canvas
 *     resolution with the full ~440k-Gaussian cloud.
 *   * CPU fallback: the Rust software rasterizer inside WASM (canvas 2D),
 *     with the quality slider scaling the internal resolution.
 *
 * Orbit camera + clip-time scrubbing; detection cuboids, the ego trajectory
 * and camera frustums are drawn as overlays on both paths.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import type { Foundation4D } from '@/lib/f4d';
import {
  WebGpuSplatRenderer,
  parseTracks,
  type FrameParams,
  type TrackParsed,
} from '@/lib/webgpu-splat';

interface Props {
  f4d: Foundation4D | null;
  ready: boolean;
  tauRef: { current: number };
  playing: boolean;
  overlays: number;
  quality: number; // internal resolution scale (CPU path)
  renderer: 'auto' | 'cpu';
  onFps?: (fps: number) => void;
  onRenderer?: (mode: 'webgpu' | 'cpu') => void;
}

const NF = 32;

export function ReconView({ f4d, ready, tauRef, playing, overlays, quality, renderer, onFps, onRenderer }: Props) {
  const gpuCanvasRef = useRef<HTMLCanvasElement>(null);
  const cpuCanvasRef = useRef<HTMLCanvasElement>(null);
  const cam = useRef({ yaw: 0.85, pitch: 0.45, dist: 28 });
  const playingRef = useRef(playing);
  const overlaysRef = useRef(overlays);
  const qualityRef = useRef(quality);
  const rendererRef = useRef(renderer);
  const onFpsRef = useRef(onFps);
  const onRendererRef = useRef(onRenderer);
  const gpuRef = useRef<WebGpuSplatRenderer | null>(null);
  const gpuFailedRef = useRef(false);
  const tracksRef = useRef<TrackParsed[]>([]);
  const rigRef = useRef<Float32Array>(new Float32Array());
  const trajRef = useRef<Float32Array>(new Float32Array());
  const [gpuActive, setGpuActive] = useState(false);
  useEffect(() => { onFpsRef.current = onFps; }, [onFps]);
  useEffect(() => { onRendererRef.current = onRenderer; }, [onRenderer]);
  const needRender = useRef(true);
  const drag = useRef<{ x: number; y: number } | null>(null);

  useEffect(() => { playingRef.current = playing; needRender.current = true; }, [playing]);
  useEffect(() => { overlaysRef.current = overlays; needRender.current = true; }, [overlays]);
  useEffect(() => { qualityRef.current = quality; needRender.current = true; }, [quality]);
  useEffect(() => { rendererRef.current = renderer; needRender.current = true; }, [renderer]);

  const computeEyeFwd = useCallback((traj: Float32Array, t: number) => {
    // Orbit target: the assembled (PGO) ego position near clip time t.
    const fi = Math.min(NF - 1, Math.max(0, Math.round(t / 0.4)));
    const base = 2 * NF * 7 + fi * 7;
    const tx = traj[base + 4];
    const ty = traj[base + 5];
    const prev = fi > 0 ? 2 * NF * 7 + (fi - 1) * 7 : base;
    const heading = Math.atan2(ty - traj[prev + 5], tx - traj[prev + 4]);
    const c = cam.current;
    const oy = (Number.isFinite(heading) ? heading : 0) + c.yaw;
    const eye: [number, number, number] = [
      tx - Math.cos(oy) * Math.cos(c.pitch) * c.dist,
      ty - Math.sin(oy) * Math.cos(c.pitch) * c.dist,
      Math.max(0.8, 0.6 + Math.sin(c.pitch) * c.dist),
    ];
    const fwd: [number, number, number] = [tx - eye[0], ty - eye[1], 0.4 - eye[2]];
    return { eye, fwd };
  }, []);

  // ---- WebGPU renderer lifecycle: create after the pipeline completes,
  // destroy + recreate on every re-run (new Gaussian cloud). ----
  useEffect(() => {
    if (!f4d || !ready || rendererRef.current === 'cpu' || gpuFailedRef.current) return;
    let cancelled = false;
    // Tear down any previous renderer first (re-run case).
    gpuRef.current?.destroy();
    gpuRef.current = null;
    setGpuActive(false);
    (async () => {
      const canvas = gpuCanvasRef.current;
      if (!canvas) return;
      const gpu = await WebGpuSplatRenderer.create(canvas);
      if (!gpu || cancelled) {
        if (gpu) gpu.destroy();
        gpuFailedRef.current = true;
        onRendererRef.current?.('cpu');
        return;
      }
      const n = f4d.gaussianCount();
      if (n === 0) { gpu.destroy(); return; }
      // Pack colors: rgb + opacity into one u32 per Gaussian.
      const color = f4d.gaussColor();
      const opacity = f4d.gaussOpacity();
      const packed = new Uint32Array(n);
      for (let i = 0; i < n; i++) {
        const a = Math.max(0, Math.min(255, Math.round(opacity[i] * 255)));
        packed[i] = color[i * 3] | (color[i * 3 + 1] << 8) | (color[i * 3 + 2] << 16) | (a << 24);
      }
      // Pack the sky cubemap (6 x 16x16 RGB) into u32s.
      const sky = f4d.sky();
      const skyPacked = new Uint32Array(sky.length / 3);
      for (let i = 0; i < skyPacked.length; i++) {
        skyPacked[i] = sky[i * 3] | (sky[i * 3 + 1] << 8) | (sky[i * 3 + 2] << 16);
      }
      gpu.uploadScene({
        n,
        pos: f4d.gaussPos(),
        quat: f4d.gaussQuat(),
        scale: f4d.gaussScale(),
        colorPacked: packed,
        skyPacked,
      });
      gpu.uploadActorIds(f4d.gaussActor());
      tracksRef.current = parseTracks(f4d.tracks());
      rigRef.current = f4d.rig();
      const traj = f4d.trajectories();
      trajRef.current = traj.slice(2 * NF * 7, 3 * NF * 7); // assembled (PGO) block
      if (cancelled) { gpu.destroy(); return; }
      gpuRef.current = gpu;
      setGpuActive(true);
      onRendererRef.current?.('webgpu');
      needRender.current = true;
    })();
    return () => { cancelled = true; };
  }, [f4d, ready]);

  useEffect(() => {
    return () => {
      gpuRef.current?.destroy();
      gpuRef.current = null;
    };
  }, []);

  // ---- Render loop (both paths). ----
  useEffect(() => {
    if (!f4d || !ready) return;
    let raf = 0;
    let lastFpsT = performance.now();
    let frames = 0;
    let lastFrameT = performance.now();
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const tau = tauRef.current;
      const gpu = gpuRef.current;
      const useGpu = !!gpu && rendererRef.current !== 'cpu';
      const activeCanvas = useGpu ? gpuCanvasRef.current : cpuCanvasRef.current;
      if (!activeCanvas) return;
      if (!playingRef.current && !needRender.current) return;
      needRender.current = false;
      const cw = Math.max(320, activeCanvas.clientWidth);
      const ch = Math.max(200, activeCanvas.clientHeight);
      if (useGpu) {
        // WebGPU path: full resolution (device pixels, capped at 2x DPR).
        const dpr = Math.min(window.devicePixelRatio || 1, 2);
        const bw = Math.floor(cw * dpr);
        const bh = Math.floor(ch * dpr);
        if (activeCanvas.width !== bw || activeCanvas.height !== bh) {
          activeCanvas.width = bw;
          activeCanvas.height = bh;
        }
        const traj = trajRef.current;
        const { eye, fwd } = computeEyeFwd(traj.length ? traj : f4d.trajectories().slice(2 * NF * 7, 3 * NF * 7), tau);
        const params: FrameParams = {
          eye,
          fwd,
          fovY: 1.05,
          tau,
          overlays: overlaysRef.current,
          tracks: tracksRef.current,
          traj: trajRef.current,
          rig: rigRef.current,
        };
        gpu!.render(params, activeCanvas.width, activeCanvas.height);
        const now = performance.now();
        const ms = now - lastFrameT;
        lastFrameT = now;
        // Report the achieved frame interval as the render time / FPS stat.
        f4d.setStat(19, ms);
        f4d.setStat(44, ms > 0 ? 1000 / ms : 0);
        f4d.setStat(20, f4d.gaussianCount());
      } else {
        // CPU path: Rust software rasterizer at quality-scaled resolution.
        const qw = Math.max(320, Math.floor(cw * qualityRef.current));
        const qh = Math.max(200, Math.floor(ch * qualityRef.current));
        if (activeCanvas.width !== qw || activeCanvas.height !== qh) {
          activeCanvas.width = qw;
          activeCanvas.height = qh;
          f4d.setRenderSize(qw, qh);
        }
        const traj = f4d.trajectories();
        const { eye, fwd } = computeEyeFwd(traj, tau);
        const t0 = performance.now();
        f4d.render(eye, fwd, 1.05, tau, overlaysRef.current);
        const ms = performance.now() - t0;
        f4d.noteRender(ms);
        const px = f4d.renderPixels();
        const { w, h } = f4d.renderDims();
        if (px.length === w * h * 4 && w > 0) {
          const ctx = activeCanvas.getContext('2d');
          if (ctx) {
            const img = new ImageData(
              new Uint8ClampedArray(px.buffer.slice(px.byteOffset, px.byteOffset + px.length) as ArrayBuffer),
              w, h,
            );
            ctx.putImageData(img, 0, 0);
          }
        }
        lastFrameT = performance.now();
      }
      frames++;
      const now = performance.now();
      if (now - lastFpsT > 500) {
        onFpsRef.current?.((frames * 1000) / (now - lastFpsT));
        frames = 0;
        lastFpsT = now;
      }
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [f4d, ready, computeEyeFwd, tauRef]);

  const onPointerDown = (e: React.PointerEvent) => {
    drag.current = { x: e.clientX, y: e.clientY };
    (e.target as HTMLElement).setPointerCapture(e.pointerId);
  };
  const onPointerMove = (e: React.PointerEvent) => {
    if (!drag.current) return;
    const dx = e.clientX - drag.current.x;
    const dy = e.clientY - drag.current.y;
    drag.current = { x: e.clientX, y: e.clientY };
    cam.current.yaw += dx * 0.008;
    cam.current.pitch = Math.min(1.35, Math.max(-0.1, cam.current.pitch + dy * 0.006));
    needRender.current = true;
  };
  const onPointerUp = () => { drag.current = null; };
  const onWheel = (e: React.WheelEvent) => {
    cam.current.dist = Math.min(70, Math.max(6, cam.current.dist * (1 + Math.sign(e.deltaY) * 0.1)));
    needRender.current = true;
  };

  return (
    <div className="relative w-full h-full min-h-[420px] rounded-lg overflow-hidden bg-black touch-none">
      <canvas
        ref={gpuCanvasRef}
        className="absolute inset-0 w-full h-full cursor-grab active:cursor-grabbing"
        style={{ visibility: gpuActive ? 'visible' : 'hidden' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onWheel={onWheel}
      />
      <canvas
        ref={cpuCanvasRef}
        className="absolute inset-0 w-full h-full cursor-grab active:cursor-grabbing"
        style={{ visibility: gpuActive ? 'hidden' : 'visible' }}
        onPointerDown={onPointerDown}
        onPointerMove={onPointerMove}
        onPointerUp={onPointerUp}
        onWheel={onWheel}
      />
      {!ready && (
        <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-sm bg-black/60">
          Run the pipeline to reconstruct the scene.
        </div>
      )}
      <div className="absolute bottom-2 left-2 text-[10px] text-white/60 bg-black/40 rounded px-2 py-1 pointer-events-none">
        drag = orbit · wheel = zoom
      </div>
      <div className="absolute top-2 left-2 text-[10px] rounded px-2 py-1 pointer-events-none bg-black/40"
        style={{ color: gpuActive ? '#7ee787' : '#e3b341' }}>
        {gpuActive ? 'WebGPU splat path' : 'CPU splat path (WASM)'}
      </div>
    </div>
  );
}
