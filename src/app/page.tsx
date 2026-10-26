'use client';

/**
 * Foundation4D in Rust — interactive demo.
 *
 * arXiv:2610.30340 "Foundation4D: Towards a unified perception and
 * reconstruction model for autonomous vehicles" implemented as a dependency-
 * free Rust crate compiled to WebAssembly (SIMD128), running the full video
 * inference pipeline in the browser:
 *
 *   surround cameras -> factorized spatiotemporal attention backbone ->
 *   geometry (plane-sweep depth stand-in) -> semantic occupancy ->
 *   3D detection -> Kalman/Hungarian/RTS tracking -> pose graph optimization ->
 *   per-pixel Gaussians (Eq. 9) -> tile-based splat rendering.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import { Foundation4D } from '@/lib/f4d';
import { ReconView } from '@/components/f4d/recon-view';
import { CameraGrid } from '@/components/f4d/camera-grid';
import { BevView } from '@/components/f4d/bev-view';
import { ArchView } from '@/components/f4d/arch-view';
import { StatsPanel } from '@/components/f4d/stats-panel';
import { Badge } from '@/components/ui/badge';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';
import { Progress } from '@/components/ui/progress';
import { Separator } from '@/components/ui/separator';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';

const STAGE_LABELS = [
  'building scene', 'rendering 192 camera images', 'window predictions',
  'backbone window', 'pose graph optimization', 'SGM plane-sweep depth',
  'semantic occupancy', '3D detection', 'tracking', 'Gaussian decode',
];
const N_FRAMES = 32;
const TAU_MAX = (N_FRAMES - 1) * 0.4;

function stageLabel(s: number, nw: number): string {
  if (s < 0) return 'done';
  if (s <= 2) return STAGE_LABELS[s];
  if (s < 3 + nw) return `factorized attention — window ${s - 2}/${nw}`;
  if (s === 3 + nw) return STAGE_LABELS[4];
  if (s < 4 + nw + N_FRAMES) return `SGM plane-sweep depth — frame ${s - 3 - nw}/${N_FRAMES}`;
  return STAGE_LABELS[s - nw - N_FRAMES + 1] ?? `stage ${s}`;
}

/** stage-index -> stats timing index */
function stageStatIdx(s: number, nw: number): number {
  if (s === 0) return 22;
  if (s === 1) return 23;
  if (s === 2) return 24;
  if (s < 3 + nw) return 13; // backbone (accumulate)
  if (s === 3 + nw) return 25; // pgo
  if (s < 4 + nw + N_FRAMES) return 30; // depth (accumulate)
  if (s === 4 + nw + N_FRAMES) return 27;
  if (s === 5 + nw + N_FRAMES) return 28;
  if (s === 6 + nw + N_FRAMES) return 29;
  return 31;
}

export default function Home() {
  const [f4d, setF4d] = useState<Foundation4D | null>(null);
  const [loadErr, setLoadErr] = useState<string | null>(null);
  const [phase, setPhase] = useState<'idle' | 'running' | 'ready'>('idle');
  const [stageDone, setStageDone] = useState(0);
  const [stageTotal, setStageTotal] = useState(16);
  const [stageName, setStageName] = useState('');
  const [stats, setStats] = useState<Float32Array | null>(null);
  const [seed, setSeed] = useState(42);
  const [tab, setTab] = useState('recon');
  const [frame, setFrame] = useState(16);
  const [tau, setTau] = useState(6.0);
  const tauRef = useRef(6.0);
  const [playing, setPlaying] = useState(true);
  const [overlays, setOverlays] = useState(0b011);
  const [quality, setQuality] = useState(0.6);
  const [fps, setFps] = useState(0);
  const [renderer, setRenderer] = useState<'auto' | 'cpu'>('auto');
  const [activeRenderer, setActiveRenderer] = useState<'webgpu' | 'cpu' | 'pending'>('pending');
  const statsTick = useRef(0);
  const [camMode, setCamMode] = useState<'rgb' | 'depth' | 'gtdepth' | 'sem' | 'conf'>('rgb');
  const [showDets, setShowDets] = useState(true);
  const [showTracks, setShowTracks] = useState(true);

  useEffect(() => {
    Foundation4D.load()
      .then((f) => { setF4d(f); (window as any).__f4d = f; })
      .catch((e) => setLoadErr(String(e)));
  }, []);

  // Playback clock.
  useEffect(() => {
    if (!playing || phase !== 'ready') return;
    let raf = 0;
    let last = performance.now();
    const loop = () => {
      raf = requestAnimationFrame(loop);
      const now = performance.now();
      const dt = (now - last) / 1000;
      last = now;
      let t = tauRef.current + dt;
      if (t > TAU_MAX) t = 0;
      tauRef.current = t;
      setTau(t);
      const fr = Math.min(N_FRAMES - 1, Math.max(0, Math.round(t / 0.4)));
      setFrame(fr);
    };
    raf = requestAnimationFrame(loop);
    return () => cancelAnimationFrame(raf);
  }, [playing, phase]);

  const runPipeline = useCallback(async () => {
    if (!f4d || phase === 'running') return;
    setPhase('running');
    setStats(null);
    try {
    f4d.init(seed);
    const total = f4d.totalStages;
    setStageTotal(total);
    setStageDone(0);
    let done = 0;
    let backboneAcc = 0;
    let depthAcc = 0;
    // Drive stages incrementally so the UI can update between them.
    while (done < total) {
      const t0 = performance.now();
      const s = f4d.step();
      const ms = performance.now() - t0;
      if (s < 0) break;
      const nw = f4d.stats()[3] | 0;
      const idx = stageStatIdx(s, nw);
      if (idx === 13 || idx === 30) {
        // Accumulate the per-window backbone and per-frame depth timings.
        if (idx === 13) backboneAcc += ms; else depthAcc += ms;
        f4d.setStat(idx, idx === 13 ? backboneAcc : depthAcc);
        const st = f4d.stats();
        if (idx === 13 && st[13] > 0) f4d.setStat(15, st[14] / (st[13] / 1000));
      } else {
        f4d.setStat(idx, ms);
      }
      done = s + 1;
      setStageDone(done);
      setStageName(stageLabel(s, nw));
      setStats(f4d.stats().slice());
      // Yield to the event loop for progress rendering.
      await new Promise((r) => setTimeout(r, 0));
    }
    // Total = sum of stage timings.
    const st = f4d.stats();
    const totalMs = st.slice(22, 32).reduce((a, b) => a + b, 0);
    f4d.setStat(32, totalMs);
    setStats(f4d.stats().slice());
    // Publish trajectories for the plot.
    window.dispatchEvent(new CustomEvent('f4d-traj', { detail: { traj: f4d.trajectories() } }));
    setPhase('ready');
    } catch (err) {
      console.error('[f4d] pipeline failed', err);
      setPhase('idle');
    }
  }, [f4d, phase, seed]);

  return (
    <div className="min-h-screen bg-background text-foreground">
      <div className="max-w-[1500px] mx-auto px-4 py-6">
        {/* Header */}
        <header className="flex flex-wrap items-start justify-between gap-4 mb-6">
          <div>
            <h1 className="text-2xl font-bold tracking-tight">
              Foundation4D <span className="text-muted-foreground font-normal">in Rust</span>
            </h1>
            <p className="text-sm text-muted-foreground mt-1 max-w-2xl">
              The unified perception &amp; reconstruction pipeline of{' '}
              <a href="https://arxiv.org/abs/2610.30340" target="_blank" rel="noreferrer"
                className="underline underline-offset-2 hover:text-foreground">
                arXiv:2610.30340
              </a>{' '}
              — factorized spatiotemporal attention, multi-view geometry, semantic
              occupancy, 3D tracking, pose-graph optimization and Gaussian splatting —
              implemented as a dependency-free Rust crate compiled to WebAssembly and
              executed entirely in your browser.
            </p>
            <div className="flex flex-wrap gap-1.5 mt-2">
              <Badge variant="secondary">Rust · no external crates</Badge>
              <Badge variant="secondary">wasm32 + SIMD128</Badge>
              <Badge variant="secondary">6-cam rig · 32 frames · 192×144</Badge>
              <Badge variant="secondary">SGM plane-sweep + PGO</Badge>
              <Badge variant="secondary">WebGPU splatting</Badge>
            </div>
          </div>
          <Card className="w-full sm:w-80 shrink-0">
            <CardHeader className="pb-2">
              <CardTitle className="text-sm">Run the pipeline</CardTitle>
            </CardHeader>
            <CardContent className="space-y-3">
              <div className="flex items-center gap-2">
                <Label htmlFor="seed" className="text-xs w-10">seed</Label>
                <Input id="seed" type="number" value={seed} min={1} max={99999}
                  onChange={(e) => setSeed(Number(e.target.value) || 42)}
                  className="h-8" disabled={phase === 'running'} />
                <Button size="sm" onClick={runPipeline} disabled={phase === 'running' || !f4d}>
                  {phase === 'running' ? 'running…' : phase === 'ready' ? 're-run' : 'run'}
                </Button>
              </div>
              {(phase === 'running' || phase === 'ready') && (
                <>
                  <Progress value={(stageDone / stageTotal) * 100} className="h-1.5" />
                  <div className="text-[11px] text-muted-foreground flex justify-between">
                    <span>{phase === 'running' ? stageName : 'pipeline complete'}</span>
                    <span className="tabular-nums">{stageDone}/{stageTotal}</span>
                  </div>
                </>
              )}
              {loadErr && <div className="text-xs text-red-400">wasm load failed: {loadErr}</div>}
            </CardContent>
          </Card>
        </header>

        {/* Main layout */}
        <div className="grid lg:grid-cols-[1fr_340px] gap-4">
          <div className="min-w-0">
            <Tabs value={tab} onValueChange={setTab}>
              <TabsList className="mb-3">
                <TabsTrigger value="recon">3D Reconstruction</TabsTrigger>
                <TabsTrigger value="cameras">Cameras</TabsTrigger>
                <TabsTrigger value="bev">BEV Occupancy</TabsTrigger>
                <TabsTrigger value="arch">Architecture</TabsTrigger>
              </TabsList>

              <TabsContent value="recon" className="mt-0">
                <div className="space-y-3">
                  <div className="h-[54vh] min-h-[380px]">
                    <ReconView f4d={f4d} ready={phase === 'ready'} tauRef={tauRef}
                      playing={playing && tab === 'recon'} overlays={overlays}
                      quality={quality} renderer={renderer} onRenderer={setActiveRenderer} onFps={(v) => {
                        setFps(v);
                        statsTick.current++;
                        if (f4d && statsTick.current % 2 === 0 && phase === 'ready') {
                          setStats(f4d.stats().slice());
                        }
                      }} />
                  </div>
                  <Card>
                    <CardContent className="py-3 space-y-3">
                      <div className="flex items-center gap-3">
                        <Button size="sm" variant="outline" onClick={() => setPlaying((p) => !p)}
                          disabled={phase !== 'ready'}>
                          {playing ? 'pause' : 'play'}
                        </Button>
                        <Slider value={[tau]} min={0} max={TAU_MAX} step={0.02}
                          onValueChange={(v) => {
                            tauRef.current = v[0];
                            setTau(v[0]);
                            setFrame(Math.min(N_FRAMES - 1, Math.round(v[0] / 0.4)));
                          }}
                          disabled={phase !== 'ready'} className="flex-1" />
                        <div className="text-xs tabular-nums text-muted-foreground w-24 text-right">
                          t = {tau.toFixed(2)} s
                        </div>
                      </div>
                      <Separator />
                      <div className="flex flex-wrap gap-x-6 gap-y-2 items-center text-xs">
                        <span className="text-muted-foreground">overlays</span>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <Switch checked={(overlays & 1) !== 0}
                            onCheckedChange={(v) => setOverlays((o) => v ? o | 1 : o & ~1)} />
                          cuboids
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <Switch checked={(overlays & 2) !== 0}
                            onCheckedChange={(v) => setOverlays((o) => v ? o | 2 : o & ~2)} />
                          trajectory + rig
                        </label>
                        <label className="flex items-center gap-1.5 cursor-pointer">
                          <Switch checked={(overlays & 4) !== 0}
                            onCheckedChange={(v) => setOverlays((o) => v ? o | 4 : o & ~4)} />
                          camera frustums
                        </label>
                        <span className="ml-auto flex items-center gap-2">
                          <span className="text-muted-foreground">renderer</span>
                          <Button size="sm" variant={renderer === 'auto' ? 'default' : 'outline'}
                            onClick={() => setRenderer((r) => r === 'auto' ? 'cpu' : 'auto')}>
                            {renderer === 'auto'
                              ? (activeRenderer === 'cpu' ? 'WebGPU (unavailable)' : 'WebGPU')
                              : 'CPU (WASM)'}
                          </Button>
                        </span>
                        <span className="flex items-center gap-2">
                          <span className="text-muted-foreground">quality</span>
                          <Slider value={[quality]} min={0.4} max={1} step={0.05}
                            onValueChange={(v) => setQuality(v[0])} className="w-28" />
                          <span className="tabular-nums text-muted-foreground w-8">{(quality * 100).toFixed(0)}%</span>
                        </span>
                      </div>
                      <p className="text-[11px] text-muted-foreground">
                        Scene rendered from ~440k 3D Gaussians decoded per source pixel (Eq. 9);
                        actor-associated Gaussians follow the tracked cuboids, static ones stay
                        fixed; the residual transmittance is filled by the sky cubemap (Eq. 10).
                        The WebGPU path re-sorts and re-splats the full cloud on the GPU every
                        frame — 60 fps at full resolution on a discrete GPU — while the CPU path
                        runs the Rust software rasterizer in WASM (quality slider scales it).
                      </p>
                    </CardContent>
                  </Card>
                </div>
              </TabsContent>

              <TabsContent value="cameras" className="mt-0">
                <div className="space-y-3">
                  <Card>
                    <CardContent className="py-3 flex flex-wrap items-center gap-2">
                      {(['rgb', 'depth', 'gtdepth', 'sem', 'conf'] as const).map((m) => (
                        <Button key={m} size="sm" variant={camMode === m ? 'default' : 'outline'}
                          onClick={() => setCamMode(m)} disabled={phase !== 'ready'}>
                          {m === 'rgb' ? 'input RGB' : m === 'depth' ? 'depth (predicted)' :
                            m === 'gtdepth' ? 'depth (GT)' : m === 'sem' ? 'semantics (predicted)' : 'confidence'}
                        </Button>
                      ))}
                      <span className="ml-auto text-xs text-muted-foreground">
                        frame {frame} / {N_FRAMES - 1} · t = {(frame * 0.4).toFixed(1)} s
                      </span>
                      <Slider value={[frame]} min={0} max={N_FRAMES - 1} step={1}
                        onValueChange={(v) => {
                          setFrame(v[0]);
                          tauRef.current = v[0] * 0.4;
                          setTau(v[0] * 0.4);
                        }}
                        disabled={phase !== 'ready'} className="w-40" />
                    </CardContent>
                  </Card>
                  <CameraGrid f4d={f4d} ready={phase === 'ready'} frame={frame} mode={camMode} />
                  <p className="text-[11px] text-muted-foreground">
                    Depth from a coarse-to-fine plane sweep with SGM cost aggregation over
                    the camera ring and temporal neighbors — the classical stand-in for the
                    trained DPT geometry decoder; confidence is the aggregated cost-curve
                    curvature. The rig&apos;s near-field blind strip uses a flat-ground
                    monocular prior.
                  </p>
                </div>
              </TabsContent>

              <TabsContent value="bev" className="mt-0">
                <div className="space-y-3">
                  <Card>
                    <CardContent className="py-3 flex flex-wrap items-center gap-4 text-xs">
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <Switch checked={showDets} onCheckedChange={setShowDets} /> detections
                      </label>
                      <label className="flex items-center gap-1.5 cursor-pointer">
                        <Switch checked={showTracks} onCheckedChange={setShowTracks} /> tracks (RTS-smoothed)
                      </label>
                      <span className="ml-auto text-muted-foreground">
                        frame {frame} / {N_FRAMES - 1}
                      </span>
                      <Slider value={[frame]} min={0} max={N_FRAMES - 1} step={1}
                        onValueChange={(v) => {
                          setFrame(v[0]);
                          tauRef.current = v[0] * 0.4;
                          setTau(v[0] * 0.4);
                        }}
                        disabled={phase !== 'ready'} className="w-40" />
                    </CardContent>
                  </Card>
                  <BevView f4d={f4d} ready={phase === 'ready'} frame={frame}
                    showDets={showDets} showTracks={showTracks} />
                  <p className="text-[11px] text-muted-foreground">
                    0.5 m voxel grid (240 x 120 x 16 over the paper&apos;s driving ROI), fused
                    per timestamp from the six cameras&apos; depth + semantics by ray marching.
                    Detections shown come from the annotation-style cuboid input of the tracker;
                    the BEV-clustering stand-in head runs alongside in the Rust code.
                  </p>
                </div>
              </TabsContent>

              <TabsContent value="arch" className="mt-0">
                <ArchView f4d={f4d} ready={phase === 'ready'} />
              </TabsContent>
            </Tabs>
          </div>

          {/* Sidebar */}
          <aside className="space-y-3">
            <StatsPanel stats={stats as any} fps={fps} />
          </aside>
        </div>

        <footer className="mt-8 pt-4 border-t border-border text-[11px] text-muted-foreground space-y-1">
          <p>
            Implementation notes: the factorized attention backbone runs a real forward pass
            with random-initialized weights (architecture + measured compute stand in for the
            trained DINOv2/VGGT-scale model). Geometry, occupancy, tracking, pose-graph
            optimization and Gaussian reconstruction are faithful implementations of the
            paper&apos;s algorithms at demo scale (192×144 cameras, 32-frame clip processed as
            8-frame overlapping windows, 0.5 m voxels, SGM-aggregated plane-sweep depth to
            90 m, ~440k Gaussians splatted through WebGPU or the WASM software rasterizer).
          </p>
        </footer>
      </div>
    </div>
  );
}
