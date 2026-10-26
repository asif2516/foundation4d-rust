'use client';

/**
 * BEV occupancy view: the per-timestamp semantic occupancy grid (240x120 at
 * 0.5 m voxels, fused from the multi-view depth predictions), with raw
 * detections and smoothed tracks overlaid.
 */

import { useEffect, useRef } from 'react';
import { F4D, type Foundation4D } from '@/lib/f4d';

const X_MIN = -20, Y_MIN = -30, VOXEL = 0.5;

const SEM_COLORS: string[] = [
  '#3a3a40', '#55555c', '#9c8f78', '#be9a6e',
  '#469650', '#e67857', '#f9be5a', '#8cc8f0',
];

interface TrackRec {
  id: number;
  cls: number;
  dims: [number, number, number];
  states: [number, number, number, number, number, number, number][]; // t, x, y, vx, vy, yaw
}

export function parseTracks(buf: Float32Array): TrackRec[] {
  const out: TrackRec[] = [];
  if (buf.length < 1) return out;
  const n = buf[0];
  let i = 1;
  for (let k = 0; k < n; k++) {
    const id = buf[i];
    const cls = buf[i + 1];
    const dims: [number, number, number] = [buf[i + 2], buf[i + 3], buf[i + 4]];
    const ns = buf[i + 5];
    i += 6;
    const states: [number, number, number, number, number, number, number][] = [];
    for (let s = 0; s < ns; s++) {
      states.push([buf[i], buf[i + 1], buf[i + 2], buf[i + 3], buf[i + 4], buf[i + 5], buf[i + 5 + 0]]);
      i += 6;
    }
    out.push({ id, cls, dims, states });
  }
  return out;
}

export function BevView({ f4d, ready, frame, showDets, showTracks }: {
  f4d: Foundation4D | null;
  ready: boolean;
  frame: number;
  showDets: boolean;
  showTracks: boolean;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!f4d || !ready) return;
    const canvas = canvasRef.current;
    if (!canvas) return;
    const W = F4D.BEV_NX, H = F4D.BEV_NY;
    canvas.width = W;
    canvas.height = H;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.fillStyle = '#101014';
    ctx.fillRect(0, 0, W, H);

    const cls = f4d.bevClass(frame);
    const height = f4d.bevHeight(frame);
    for (let y = 0; y < H; y++) {
      for (let x = 0; x < W; x++) {
        const b = y * W + x;
        const c = cls[b];
        if (c === 0) continue;
        ctx.fillStyle = SEM_COLORS[c] ?? '#3a3a40';
        ctx.fillRect(x, y, 1, 1);
        const h = height[b];
        if (Number.isFinite(h) && h > 3.0) {
          // Tall structure: darken toward building.
          ctx.fillStyle = 'rgba(190,154,110,0.5)';
          ctx.fillRect(x, y, 1, 1);
        }
      }
    }

    // Ego marker at the frame's rig position (grid is clip-frame, ego at
    // frame t sits near x = t*3.2 approximately — use GT trajectory).
    const traj = f4d.trajectories();
    const NF = 32;
    const base = 3 * NF * 7 + frame * 7; // GT block
    const ex = (traj[base + 4] - X_MIN) / VOXEL;
    const ey = (traj[base + 5] - Y_MIN) / VOXEL;
    ctx.fillStyle = '#ffffff';
    ctx.fillRect(ex - 3, ey - 2, 6, 4);
    ctx.fillStyle = '#f4d34d';
    ctx.fillRect(ex - 1, ey - 1, 2, 2);

    if (showTracks) {
      const tracks = parseTracks(f4d.tracks());
      for (const tr of tracks) {
        ctx.strokeStyle = tr.cls === 6 ? '#ff9c5a' : '#57d98a';
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        let started = false;
        for (const s of tr.states) {
          const x = (s[1] - X_MIN) / VOXEL;
          const y = (s[2] - Y_MIN) / VOXEL;
          if (!started) { ctx.moveTo(x, y); started = true; } else { ctx.lineTo(x, y); }
        }
        ctx.stroke();
      }
    }

    if (showDets) {
      const dets = f4d.detections(frame);
      const n = dets.length > 0 ? dets[0] : 0;
      for (let k = 0; k < n; k++) {
        const o = 1 + k * 11;
        const cx = (dets[o] - X_MIN) / VOXEL;
        const cy = (dets[o + 1] - Y_MIN) / VOXEL;
        const l = dets[o + 3] / VOXEL;
        const w = dets[o + 4] / VOXEL;
        const yaw = dets[o + 6];
        ctx.save();
        ctx.translate(cx, cy);
        ctx.rotate(yaw);
        ctx.strokeStyle = dets[o + 7] === 6 ? '#ff9c5a' : '#57d98a';
        ctx.lineWidth = 1;
        ctx.strokeRect(-l / 2, -w / 2, l, w);
        // Heading tick.
        ctx.beginPath();
        ctx.moveTo(l / 2, 0);
        ctx.lineTo(l / 2 - 3, -2);
        ctx.stroke();
        ctx.restore();
      }
    }
  }, [f4d, ready, frame, showDets, showTracks]);

  return (
    <div className="w-full">
      <canvas
        ref={canvasRef}
        className="w-full aspect-2/1 block rounded-md border border-border bg-[#101014]"
        style={{ imageRendering: 'pixelated' }}
      />
      <div className="flex flex-wrap gap-2 mt-2 text-[10px] text-muted-foreground">
        {['road', 'sidewalk', 'building', 'vegetation', 'vehicle', 'person', 'sky'].map((n, i) => (
          <span key={n} className="inline-flex items-center gap-1">
            <span className="w-2.5 h-2.5 rounded-sm" style={{ background: SEM_COLORS[i + 1] }} />
            {n}
          </span>
        ))}
      </div>
    </div>
  );
}
