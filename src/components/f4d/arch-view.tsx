'use client';

/**
 * Architecture view: the factorized spatiotemporal aggregator's structure —
 * the camera x time token grid, the alternating attention groups (within-image,
 * cross-camera, cross-temporal), the special tokens, and live backbone outputs
 * (contextual feature maps projected to RGB + raw head outputs, random-init).
 */

import { useEffect, useRef } from 'react';
import { F4D, type Foundation4D } from '@/lib/f4d';

export function ArchView({ f4d, ready }: { f4d: Foundation4D | null; ready: boolean }) {
  const featRefs = useRef<(HTMLCanvasElement | null)[]>([]);
  const depthRef = useRef<HTMLCanvasElement>(null);
  const semRef = useRef<HTMLCanvasElement>(null);

  useEffect(() => {
    if (!f4d || !ready) return;
    // Feature maps: 6 cams x 16x12 RGB.
    const feats = f4d.features();
    for (let c = 0; c < 6; c++) {
      const canvas = featRefs.current[c];
      if (!canvas) continue;
      const ctx = canvas.getContext('2d');
      if (!ctx) continue;
      canvas.width = F4D.GW;
      canvas.height = F4D.GH;
      const img = ctx.createImageData(F4D.GW, F4D.GH);
      for (let p = 0; p < F4D.GW * F4D.GH; p++) {
        const o = (c * F4D.GW * F4D.GH + p) * 3;
        img.data[p * 4] = feats[o];
        img.data[p * 4 + 1] = feats[o + 1];
        img.data[p * 4 + 2] = feats[o + 2];
        img.data[p * 4 + 3] = 255;
      }
      ctx.putImageData(img, 0, 0);
    }
    // Head outputs for the front camera (random-init demonstration).
    const hd = f4d.headDepth();
    const d0 = depthRef.current;
    if (d0 && hd.length > 0) {
      const ctx = d0.getContext('2d');
      if (ctx) {
        d0.width = F4D.GW;
        d0.height = F4D.GH;
        const img = ctx.createImageData(F4D.GW, F4D.GH);
        for (let p = 0; p < F4D.GW * F4D.GH; p++) {
          const v = Math.min(1, Math.max(0, hd[p * 2] * 0.5 + 0.5));
          img.data[p * 4] = 60 + 195 * v;
          img.data[p * 4 + 1] = 60 + 140 * v;
          img.data[p * 4 + 2] = 80;
          img.data[p * 4 + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
      }
    }
    const hs = f4d.headSem();
    const s0 = semRef.current;
    if (s0 && hs.length > 0) {
      const ctx = s0.getContext('2d');
      if (ctx) {
        s0.width = F4D.GW;
        s0.height = F4D.GH;
        const img = ctx.createImageData(F4D.GW, F4D.GH);
        for (let p = 0; p < F4D.GW * F4D.GH; p++) {
          // Argmax class -> hue.
          let best = 0, bv = -1e9;
          for (let k = 0; k < 8; k++) {
            if (hs[p * 8 + k] > bv) { bv = hs[p * 8 + k]; best = k; }
          }
          const hue = (best * 45) % 360;
          img.data[p * 4] = 80 + (hue % 3) * 40;
          img.data[p * 4 + 1] = 60 + ((hue / 3 | 0) % 3) * 40;
          img.data[p * 4 + 2] = 90 + ((hue / 9 | 0) % 3) * 40;
          img.data[p * 4 + 3] = 255;
        }
        ctx.putImageData(img, 0, 0);
      }
    }
  }, [f4d, ready]);

  const camNames = ['front', 'left-front', 'left-rear', 'rear', 'right-rear', 'right-front'];

  return (
    <div className="space-y-4">
      {/* Factorization diagram. */}
      <div className="rounded-lg border border-border p-4 bg-card">
        <h3 className="text-sm font-semibold mb-3">Factorized spatiotemporal attention (paper Sec 3.3)</h3>
        <svg viewBox="0 0 640 300" className="w-full">
          {/* Camera x time token grid: 6 cams x 4 frames. */}
          {Array.from({ length: 4 }).map((_, t) =>
            Array.from({ length: 6 }).map((_, c) => {
              const x = 40 + c * 90;
              const y = 40 + t * 60;
              return (
                <g key={`${t}-${c}`}>
                  <rect x={x} y={y} width={70} height={40} rx={4}
                    fill={c === 0 ? '#1d3a2f' : '#16211c'} stroke="#3a4a42" />
                  <text x={x + 35} y={y + 24} textAnchor="middle" fontSize={10} fill="#8fa89b">
                    {`${t},${c}`}
                  </text>
                </g>
              );
            }),
          )}
          {/* Within-image attention: rounded rect inside each image. */}
          {Array.from({ length: 4 }).map((_, t) =>
            Array.from({ length: 6 }).map((_, c) => {
              const x = 40 + c * 90;
              const y = 40 + t * 60;
              return <rect key={`wi-${t}-${c}`} x={x + 5} y={y + 5} width={60} height={30} rx={8}
                fill="none" stroke="#57d98a" strokeWidth={1.4} strokeDasharray="3 2" />;
            }),
          )}
          {/* Camera groups (fixed t): horizontal arcs. */}
          {Array.from({ length: 4 }).map((_, t) => (
            <path key={`cg-${t}`} d={`M 45 ${40 + t * 60 + 20} Q 370 ${40 + t * 60 - 18} 695 ${40 + t * 60 + 20}`}
              fill="none" stroke="#f4d34d" strokeWidth={1.6} />
          ))}
          {/* Temporal groups (fixed c): vertical lines. */}
          {Array.from({ length: 6 }).map((_, c) => (
            <line key={`tg-${c}`} x1={75 + c * 90} y1={45} x2={75 + c * 90} y2={255}
              stroke="#e67857" strokeWidth={1.6} />
          ))}
          {/* Legend. */}
          <g transform="translate(40, 272)">
            <line x1={0} y1={0} x2={22} y2={0} stroke="#57d98a" strokeWidth={1.4} strokeDasharray="3 2" />
            <text x={28} y={3} fontSize={10} fill="#9ca3af">within-image</text>
            <line x1={110} y1={0} x2={132} y2={0} stroke="#f4d34d" strokeWidth={1.6} />
            <text x={138} y={3} fontSize={10} fill="#9ca3af">camera group (fixed t)</text>
            <line x1={268} y1={0} x2={290} y2={0} stroke="#e67857" strokeWidth={1.6} />
            <text x={296} y={3} fontSize={10} fill="#9ca3af">temporal group (fixed c)</text>
            <text x={430} y={3} fontSize={10} fill="#6b7280">alternating across layers</text>
          </g>
        </svg>
        <p className="text-xs text-muted-foreground mt-2">
          Each image contributes patch tokens plus a rig token, an ego token and four register tokens.
          Blocks alternate: within-image self-attention on the pooled aggregation stream, then
          cross-image attention over camera groups or temporal groups — replacing VGGT&apos;s global
          attention across all images at once. The final representation concatenates the two streams.
        </p>
      </div>

      {/* Backbone outputs. */}
      <div className="rounded-lg border border-border p-4 bg-card">
        <h3 className="text-sm font-semibold mb-1">Contextual features (middle window, projected to RGB)</h3>
        <p className="text-xs text-muted-foreground mb-3">
          Random-initialized weights — the forward pass is real (measured compute below), standing in
          for the trained DINOv2/VGGT-scale parameters.
        </p>
        <div className="grid grid-cols-3 md:grid-cols-6 gap-2">
          {camNames.map((n, c) => (
            <div key={c}>
              <canvas ref={(el) => { featRefs.current[c] = el; }}
                className="w-full aspect-4/3 rounded bg-black block"
                style={{ imageRendering: 'pixelated' }} />
              <div className="text-[10px] text-muted-foreground font-mono text-center">{n}</div>
            </div>
          ))}
        </div>
      </div>

      <div className="grid md:grid-cols-2 gap-4">
        <div className="rounded-lg border border-border p-4 bg-card">
          <h3 className="text-sm font-semibold mb-1">Depth head output (random init)</h3>
          <canvas ref={depthRef} className="w-full aspect-4/3 rounded bg-black block"
            style={{ imageRendering: 'pixelated' }} />
        </div>
        <div className="rounded-lg border border-border p-4 bg-card">
          <h3 className="text-sm font-semibold mb-1">Semantic head logits (random init)</h3>
          <canvas ref={semRef} className="w-full aspect-4/3 rounded bg-black block"
            style={{ imageRendering: 'pixelated' }} />
        </div>
      </div>
    </div>
  );
}
