'use client';

/**
 * Surround-camera view: the six rendered rig cameras with switchable overlays —
 * RGB input, predicted depth (plane-sweep stand-in for the geometry decoder),
 * GT depth, predicted semantics, and stereo confidence.
 */

import { useEffect, useRef } from 'react';
import { F4D, type Foundation4D } from '@/lib/f4d';

type Mode = 'rgb' | 'depth' | 'gtdepth' | 'sem' | 'conf';

const CAM_NAMES = ['front', 'left-front', 'left-rear', 'rear', 'right-rear', 'right-front'];

const SEM_COLORS: [number, number, number][] = [
  [90, 90, 90], [80, 80, 86], [150, 140, 120], [190, 150, 110],
  [70, 150, 80], [230, 120, 90], [250, 190, 90], [140, 200, 240],
];

function depthColor(d: number, maxD = 60): [number, number, number] {
  const t = Math.min(1, Math.max(0, d / maxD));
  // Turbo-ish ramp.
  const r = Math.min(255, 255 * (t < 0.25 ? 4 * t : 1));
  const g = Math.min(255, 255 * (t > 0.3 ? Math.max(0, 1 - Math.abs(t - 0.55) * 2.6) : t * 3));
  const b = Math.min(255, 255 * Math.max(0, 1 - t * 1.8));
  return [r, g, b];
}

export function CameraGrid({ f4d, ready, frame, mode }: {
  f4d: Foundation4D | null;
  ready: boolean;
  frame: number;
  mode: Mode;
}) {
  const refs = useRef<(HTMLCanvasElement | null)[]>([]);

  useEffect(() => {
    if (!f4d || !ready) return;
    for (let c = 0; c < 6; c++) {
      const canvas = refs.current[c];
      if (!canvas) continue;
      const ctx = canvas.getContext('2d');
      if (!ctx) continue;
      canvas.width = F4D.IMG_W * 2;
      canvas.height = F4D.IMG_H * 2;
      const img = ctx.createImageData(F4D.IMG_W, F4D.IMG_H);
      const px = img.data;
      if (mode === 'rgb') {
        const rgb = f4d.image(frame, c);
        for (let p = 0; p < F4D.IMG_W * F4D.IMG_H; p++) {
          px[p * 4] = rgb[p * 3];
          px[p * 4 + 1] = rgb[p * 3 + 1];
          px[p * 4 + 2] = rgb[p * 3 + 2];
          px[p * 4 + 3] = 255;
        }
      } else if (mode === 'depth' || mode === 'gtdepth') {
        const d = mode === 'depth' ? f4d.predDepth(frame, c) : f4d.gtDepth(frame, c);
        for (let p = 0; p < F4D.IMG_W * F4D.IMG_H; p++) {
          const [r, g, b] = depthColor(d[p]);
          px[p * 4] = r;
          px[p * 4 + 1] = g;
          px[p * 4 + 2] = b;
          px[p * 4 + 3] = 255;
        }
      } else if (mode === 'sem') {
        const s = f4d.predSem(frame, c);
        for (let p = 0; p < F4D.IMG_W * F4D.IMG_H; p++) {
          const col = SEM_COLORS[s[p]] ?? [60, 60, 60];
          px[p * 4] = col[0];
          px[p * 4 + 1] = col[1];
          px[p * 4 + 2] = col[2];
          px[p * 4 + 3] = 255;
        }
      } else {
        const cf = f4d.predConf(frame, c);
        for (let p = 0; p < F4D.IMG_W * F4D.IMG_H; p++) {
          const v = Math.min(1, Math.max(0, cf[p]));
          px[p * 4] = 40 + 210 * (1 - v);
          px[p * 4 + 1] = 40 + 200 * v;
          px[p * 4 + 2] = 60;
          px[p * 4 + 3] = 255;
        }
      }
      ctx.putImageData(img, 0, 0);
      // Upscale.
      const tmp = document.createElement('canvas');
      tmp.width = F4D.IMG_W;
      tmp.height = F4D.IMG_H;
      tmp.getContext('2d')?.putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = false;
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(tmp, 0, 0, canvas.width, canvas.height);
    }
  }, [f4d, ready, frame, mode]);

  return (
    <div className="grid grid-cols-2 md:grid-cols-3 gap-2">
      {CAM_NAMES.map((name, c) => (
        <div key={c} className="rounded-md overflow-hidden border border-border">
          <canvas
            ref={(el) => { refs.current[c] = el; }}
            className="w-full aspect-4/3 block bg-black"
          />
          <div className="text-[10px] px-1.5 py-1 text-muted-foreground font-mono">{name}</div>
        </div>
      ))}
    </div>
  );
}
