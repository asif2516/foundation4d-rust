/**
 * CDP driver for the Foundation4D demo (bun run scripts/cdp.ts).
 * Drives the headless Chrome started with --remote-debugging-port=9222
 * (launched with WebGPU SwiftShader flags), runs the pipeline, verifies the
 * render path, and captures PNG frames for the demo GIF.
 */

const CDP = 'http://127.0.0.1:9222';
const OUT = process.env.OUT_DIR || '/tmp/f4d_gif';
const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms));

interface CdpMsg { id?: number; method?: string; params?: any; result?: any; error?: any }

async function main() {
  const targets = (await (await fetch(`${CDP}/json/list`)).json()) as any[];
  let page = targets.find((t) => t.type === 'page');
  if (!page) throw new Error('no page target');
  const ws = new WebSocket(page.webSocketDebuggerUrl);
  let nextId = 0;
  const pending = new Map<number, { res: (v: any) => void; rej: (e: any) => void }>();
  ws.onmessage = (e: MessageEvent) => {
    const m = JSON.parse(String(e.data)) as CdpMsg;
    if (m.id && pending.has(m.id)) {
      const p = pending.get(m.id)!;
      pending.delete(m.id);
      if (m.error) p.rej(new Error(JSON.stringify(m.error)));
      else p.res(m.result);
    }
  };
  await new Promise<void>((res, rej) => {
    ws.onopen = () => res();
    ws.onerror = () => rej(new Error('ws error'));
    setTimeout(() => rej(new Error('ws open timeout')), 10000);
  });
  const send = (method: string, params: any = {}): Promise<any> =>
    new Promise((res, rej) => {
      const id = ++nextId;
      pending.set(id, { res, rej });
      ws.send(JSON.stringify({ id, method, params }));
    });
  const ev = async (expr: string): Promise<any> =>
    (await send('Runtime.evaluate', { expression: expr, awaitPromise: true, returnByValue: true })).result.value;

  await send('Page.enable');
  await send('Runtime.enable');
  await send('Emulation.setDeviceMetricsOverride', { width: 1440, height: 900, deviceScaleFactor: 1, mobile: false });

  console.log('navigating…');
  await send('Page.navigate', { url: 'http://localhost:3000/' });
  await sleep(9000);

  // Basic sanity: page + wasm loaded.
  const loaded = await ev('typeof window.__f4d');
  console.log('f4d loaded:', loaded);
  if (loaded !== 'object') throw new Error('wasm not loaded');

  // Click "run".
  await ev(`(() => { const b = [...document.querySelectorAll('button')].find(b => b.textContent?.trim() === 'run'); b?.click(); return !!b; })()`);
  console.log('pipeline started; polling…');

  const t0 = Date.now();
  let done = 0, total = 0;
  for (;;) {
    await sleep(3000);
    const st = await ev('window.__f4d ? window.__f4d.stats()[0] + "/" + window.__f4d.totalStages : "x"');
    const m = /^(\d+)\/(\d+)$/.exec(String(st));
    if (m) { done = +m[1]; total = +m[2]; }
    console.log(`  stage ${st} (${((Date.now() - t0) / 1000).toFixed(0)}s)`);
    if (m && +m[1] >= +m[2]) break;
    if (Date.now() - t0 > 300000) throw new Error('pipeline timeout');
  }
  console.log(`pipeline complete in ${((Date.now() - t0) / 1000).toFixed(0)}s`);

  await sleep(4000); // let the GPU renderer init + first frames render
  const badge = await ev(`document.querySelector('.absolute.top-2.left-2')?.textContent || 'none'`);
  console.log('renderer badge:', badge);
  const gcount = await ev('window.__f4d.gaussianCount()');
  const stats = await ev('JSON.stringify([...window.__f4d.stats().slice(0,50)])');
  console.log('gaussians:', gcount);
  console.log('stats:', stats);

  // Collect console errors.
  const errs = await ev(`window.__f4dErrors || 'no-capture'`);
  console.log('page errors:', errs);

  // ---- Capture frames for the GIF ----
  const fs = await import('node:fs');
  fs.mkdirSync(OUT, { recursive: true });
  const shot = async (name: string) => {
    const r = await send('Page.captureScreenshot', { format: 'png' });
    fs.writeFileSync(`${OUT}/${name}`, Buffer.from(r.data, 'base64'));
  };
  // Wait for playback to be mid-clip, then capture the recon view animating.
  await sleep(2000);
  let n = 0;
  for (let i = 0; i < 26; i++) {
    await shot(`recon_${String(n++).padStart(3, '0')}.png`);
    await sleep(450);
  }
  // Orbit: drag across the canvas.
  const drag = async (dx: number, dy: number) => {
    await ev(`(() => {
      const c = [...document.querySelectorAll('canvas')].find(c => c.clientWidth > 300) || document.querySelector('canvas');
      const r = c.getBoundingClientRect();
      const opts = { bubbles: true, cancelable: true, pointerId: 1, isPrimary: true, pointerType: 'mouse' };
      c.dispatchEvent(new PointerEvent('pointerdown', { ...opts, clientX: r.x + r.width / 2, clientY: r.y + r.height / 2 }));
      c.dispatchEvent(new PointerEvent('pointermove', { ...opts, clientX: r.x + r.width / 2 + ${dx}, clientY: r.y + r.height / 2 + ${dy} }));
      c.dispatchEvent(new PointerEvent('pointerup', opts));
      return true;
    })()`);
  };
  for (let k = 0; k < 3; k++) {
    await drag(120, -20);
    for (let i = 0; i < 4; i++) {
      await shot(`recon_${String(n++).padStart(3, '0')}.png`);
      await sleep(450);
    }
  }
  // Cameras tab.
  await ev(`([...document.querySelectorAll('[role=tab]')].find(t => t.textContent?.includes('Cameras'))?.click(), true)`);
  await sleep(1500);
  await shot('cameras.png');
  await ev(`([...document.querySelectorAll('button')].find(b => b.textContent?.includes('depth (predicted)'))?.click(), true)`);
  await sleep(700);
  await shot('cameras_depth.png');
  // BEV tab.
  await ev(`([...document.querySelectorAll('[role=tab]')].find(t => t.textContent?.includes('BEV'))?.click(), true)`);
  await sleep(1500);
  await shot('bev.png');
  // Architecture tab.
  await ev(`([...document.querySelectorAll('[role=tab]')].find(t => t.textContent?.includes('Architecture'))?.click(), true)`);
  await sleep(1200);
  await shot('arch.png');
  // Back to recon for a final animated segment.
  await ev(`([...document.querySelectorAll('[role=tab]')].find(t => t.textContent?.includes('3D Reconstruction'))?.click(), true)`);
  await sleep(1500);
  for (let i = 0; i < 8; i++) {
    await shot(`recon_${String(n++).padStart(3, '0')}.png`);
    await sleep(450);
  }
  console.log(`captured ${n + 3} frames -> ${OUT}`);
  ws.close();
}

main().catch((e) => { console.error(e); process.exit(1); });
