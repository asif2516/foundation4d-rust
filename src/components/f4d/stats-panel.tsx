'use client';

import { useEffect } from 'react';

/**
 * Metrics + trajectory panel: quality metrics, per-stage timings, the
 * raw / stitched / PGO / GT ego trajectories, and the compute profile of the
 * factorized attention backbone.
 */

import type { F4DStats } from '@/lib/f4d';

const NF = 32;

function Metric({ label, value, unit, hint }: { label: string; value: string; unit?: string; hint?: string }) {
  return (
    <div className="rounded-lg border border-border bg-card p-3">
      <div className="text-[10px] uppercase tracking-wide text-muted-foreground">{label}</div>
      <div className="text-xl font-semibold tabular-nums">
        {value}
        {unit && <span className="text-xs text-muted-foreground ml-1">{unit}</span>}
      </div>
      {hint && <div className="text-[10px] text-muted-foreground mt-0.5">{hint}</div>}
    </div>
  );
}

export function StatsPanel({ stats, fps }: { stats: F4DStats | null; fps: number }) {
  if (!stats || stats[1] === 0) {
    return (
      <div className="rounded-lg border border-border p-4 text-sm text-muted-foreground">
        Metrics appear after the pipeline runs.
      </div>
    );
  }
  const done = stats[0] >= stats[1];
  const trajColors = ['#9ca3af', '#f4d34d', '#57d98a', '#ffffff'];
  void trajColors;

  const stages = [
    ['scene', stats[22]], ['render', stats[23]], ['windows', stats[24]],
    ['backbone', stats[13]], ['depth (SGM sweep)', stats[30]],
    ['pgo + stitch', stats[25]], ['occupancy', stats[27]],
    ['detection', stats[28]], ['tracking', stats[29]], ['gaussians', stats[31]],
  ] as [string, number][];

  return (
    <div className="space-y-3">
      <div className="grid grid-cols-2 gap-2">
        <Metric label="Depth AbsRel" value={done ? stats[4].toFixed(3) : '—'} hint="vs GT (non-sky)" />
        <Metric label="Depth AbsRel (far)" value={done ? stats[46].toFixed(3) : '—'} hint="GT > 30 m" />
        <Metric label="Occupancy IoU" value={done ? stats[5].toFixed(2) : '—'} hint={`sem acc ${done ? stats[6].toFixed(2) : '—'}`} />
        <Metric label="Tracks" value={done ? `${stats[8] | 0}` : '—'} hint={`${stats[9] | 0} vehicles · ${stats[10] | 0} persons`} />
        <Metric label="Gaussians" value={done ? `${(stats[11] / 1000).toFixed(0)}k` : '—'} hint="per-pixel, Eq. 9" />
        <Metric label="Clip" value={done ? `${stats[33] | 0} fr · ${stats[35] | 0}×${stats[36] | 0}` : '—'} hint={`${stats[3] | 0} windows`} />
      </div>

      <div className="rounded-lg border border-border bg-card p-3">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-2">
          Ego trajectory RMSE vs GT
        </div>
        {done && (
          <div className="space-y-1.5 text-sm">
            {[
              ['raw window chaining', stats[16]],
              ['sequential stitching', stats[17]],
              ['pose graph optimization', stats[18]],
            ].map(([n, v]) => (
              <div key={n as string} className="flex items-center gap-2">
                <div className="w-40 text-xs text-muted-foreground">{n as string}</div>
                <div className="flex-1 h-3 rounded bg-muted overflow-hidden">
                  <div className="h-full rounded bg-emerald-500/70"
                    style={{ width: `${Math.min(100, (v as number) / Math.max(0.2, stats[16]) * 100)}%` }} />
                </div>
                <div className="w-16 text-right text-xs tabular-nums">{(v as number).toFixed(2)} m</div>
              </div>
            ))}
          </div>
        )}
        <TrajectoryPlot />
      </div>

      <div className="rounded-lg border border-border bg-card p-3">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-2">
          Backbone — factorized attention
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs tabular-nums">
          <div className="flex justify-between"><span className="text-muted-foreground">compute</span><span>{stats[14].toFixed(1)} GFLOP</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">forward</span><span>{stats[13].toFixed(0)} ms</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">throughput</span><span>{stats[15].toFixed(1)} GFLOP/s</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">windows</span><span>{stats[3] | 0} x {stats[33] | 0} frames</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">width / layers / heads</span><span>{stats[37] | 0} / {stats[38] | 0} / {stats[39] | 0}</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">tokens per image</span><span>{stats[40] | 0} + {stats[41] | 0} pooled</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">camera group</span><span>{stats[42] | 0} tokens</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">temporal group</span><span>{stats[43] | 0} tokens</span></div>
        </div>
      </div>

      <div className="rounded-lg border border-border bg-card p-3">
        <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-2">
          Splat renderer
        </div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs tabular-nums">
          <div className="flex justify-between"><span className="text-muted-foreground">frame time</span><span>{stats[19].toFixed(0)} ms</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">fps (display)</span><span>{fps.toFixed(0)}</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">tile entries</span><span>{(stats[20] / 1000).toFixed(0)}k</span></div>
          <div className="flex justify-between"><span className="text-muted-foreground">blend ops</span><span>{stats[21].toFixed(1)}M</span></div>
        </div>
      </div>

      {done && (
        <div className="rounded-lg border border-border bg-card p-3">
          <div className="text-[10px] uppercase tracking-wide text-muted-foreground mb-2">
            Stage timings (JS-measured, WASM)
          </div>
          <div className="space-y-1">
            {stages.filter(([, v]) => v > 0).map(([n, v]) => (
              <div key={n} className="flex items-center gap-2 text-xs">
                <div className="w-40 text-muted-foreground">{n}</div>
                <div className="flex-1 h-2 rounded bg-muted overflow-hidden">
                  <div className="h-full rounded bg-amber-500/70"
                    style={{ width: `${Math.min(100, v / Math.max(1, stats[25] + stats[13] + stats[23]) * 100)}%` }} />
                </div>
                <div className="w-14 text-right tabular-nums">{v.toFixed(0)} ms</div>
              </div>
            ))}
            <div className="flex justify-between text-xs pt-1 border-t border-border mt-1">
              <span className="text-muted-foreground">total</span>
              <span className="tabular-nums">{stats[32].toFixed(0)} ms</span>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function TrajectoryPlot() {
  // Filled by the parent via a custom event carrying the trajectory buffer.
  const id = 'f4d-traj-plot';
  useEffect(() => {
    const handler = (e: Event) => {
      const detail = (e as CustomEvent).detail as { traj: Float32Array };
      drawPlot(document.getElementById(id) as HTMLCanvasElement, detail.traj);
    };
    window.addEventListener('f4d-traj', handler);
    return () => window.removeEventListener('f4d-traj', handler);
  }, []);
  return <canvas id={id} className="w-full aspect-16/9 rounded bg-black/40" />;
}

function drawPlot(canvas: HTMLCanvasElement, traj: Float32Array) {
  if (!canvas) return;
  const W = 480, H = 270;
  canvas.width = W;
  canvas.height = H;
  const ctx = canvas.getContext('2d');
  if (!ctx) return;
  ctx.fillStyle = '#0c0c10';
  ctx.fillRect(0, 0, W, H);
  let minX = 1e9, maxX = -1e9, minY = 1e9, maxY = -1e9;
  for (let b = 0; b < 4; b++) {
    for (let i = 0; i < NF; i++) {
      const o = b * NF * 7 + i * 7 + 4;
      minX = Math.min(minX, traj[o]); maxX = Math.max(maxX, traj[o]);
      minY = Math.min(minY, traj[o + 1]); maxY = Math.max(maxY, traj[o + 1]);
    }
  }
  const pad = 20;
  const sx = (x: number) => pad + ((x - minX) / Math.max(1e-6, maxX - minX)) * (W - 2 * pad);
  const sy = (y: number) => H - pad - ((y - minY) / Math.max(1e-6, maxY - minY)) * (H - 2 * pad);
  const colors = ['#9ca3af', '#f4d34d', '#57d98a', '#ffffff'];
  for (let b = 0; b < 4; b++) {
    ctx.strokeStyle = colors[b];
    ctx.lineWidth = b === 3 ? 2 : 1.5;
    ctx.beginPath();
    for (let i = 0; i < NF; i++) {
      const o = b * NF * 7 + i * 7 + 4;
      if (i === 0) ctx.moveTo(sx(traj[o]), sy(traj[o + 1]));
      else ctx.lineTo(sx(traj[o]), sy(traj[o + 1]));
    }
    ctx.stroke();
  }
}

