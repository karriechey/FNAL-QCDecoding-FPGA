#!/usr/bin/env python3
# Created: 2026-08-21
# Build a self-contained results page: figures/exp16_results.html
#
# One file, no network, no build step -- data is embedded as JSON and drawn with plain SVG,
# so it opens from disk and can be mailed to someone. Reads the same results_data.py the
# figures use, plus the history files pulled from grace1.
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from results_data import MWPM, SEALED, rows, load_curves, load_rcnn_curves

META = os.environ.get('RESULTS_META', '../rcnn_threshold/results_meta')
OUT = os.environ.get('FIGDIR', '../figures')

curves = []
for c in load_curves(META) + load_rcnn_curves(
        os.environ.get('RESULTS_META_EAF', os.path.join(META, '..', 'results_meta_eaf'))):
    curves.append(dict(tag=c['tag'], d=c['d'], arch=c['arch'], units=c['units'],
                       shots=c['shots'], seed=c['seed'], best=c['best'],
                       val=[round(x, 5) for x in c['val']]))

data = dict(mwpm=MWPM, sealed=SEALED, runs=rows(), curves=curves)
payload = json.dumps(data, separators=(',', ':'))

html = """<title>Experiment 16 — decoder results</title>
<style>
  :root {
    --bg:#fcfcfb; --card:#ffffff; --ink:#1a1a1a; --muted:#55534e; --line:#e6e4e0;
    --d5:#eb6834; --d7:#2a78d6; --d9:#1baf7a; --grey:#8a8782;
  }
  * { box-sizing:border-box; }
  body { margin:0; background:var(--bg); color:var(--ink);
         font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif; }
  .wrap { max-width:1440px; margin:0 auto; padding:32px 24px 64px; }
  h1 { font-size:26px; margin:0 0 6px; letter-spacing:-0.01em; }
  .sub { color:var(--muted); margin:0 0 28px; max-width:80ch; }
  .card { background:var(--card); border:1px solid var(--line); border-radius:10px;
          padding:20px 22px; margin-bottom:22px; }
  h2 { font-size:15px; text-transform:uppercase; letter-spacing:0.06em; color:var(--muted);
       margin:0 0 14px; font-weight:600; }
  .controls { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
  button { font:inherit; font-size:13px; padding:6px 13px; border-radius:999px;
           border:1px solid var(--line); background:#fff; color:var(--muted); cursor:pointer; }
  button.on { color:#fff; border-color:transparent; }
  button.on[data-k="5"] { background:var(--d5); } button.on[data-k="7"] { background:var(--d7); }
  button.on[data-k="9"] { background:var(--d9); } button.on[data-k="gru"],
  button.on[data-k="mlp"], button.on[data-k="rcnn"] { background:var(--muted); }
  svg { width:100%; height:auto; display:block; overflow:visible; }
  .tt { position:fixed; pointer-events:none; background:#1a1a1a; color:#fff; font-size:12px;
        line-height:1.45; padding:8px 10px; border-radius:6px; opacity:0; transition:opacity .1s;
        z-index:10; white-space:nowrap; }
  table { border-collapse:collapse; width:100%; font-size:13px; }
  th { text-align:right; padding:7px 9px; border-bottom:2px solid var(--line); color:var(--muted);
       font-weight:600; cursor:pointer; white-space:nowrap; }
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align:left; }
  td { padding:6px 9px; border-bottom:1px solid var(--line); white-space:nowrap; }
  tr.win td { background:#fff6f2; font-weight:600; }
  .dot { display:inline-block; width:9px; height:9px; border-radius:50%; margin-right:7px; }
  .foot { color:var(--muted); font-size:12.5px; margin-top:10px; }
  code { background:#f3f1ee; padding:1px 5px; border-radius:4px; font-size:12.5px; }
</style>
<div class="wrap">
<h1>Experiment 16 — surface-code decoding at p = 0.004</h1>
<p class="sub">Every configuration trained on grace1 (GH200), scored once on the tuning
evaluation block [15.2M, 17.0M) with MWPM decoded on those same shots. The d=5 GRU at 20M
augmented shots reaches 0.95× MWPM and holds it on the sealed block [17M, 19M).
Colour is code distance; marker shape is architecture. Hover any point.</p>

<div class="card">
  <h2>Error rate against training data</h2>
  <div class="controls" id="f1"></div>
  <svg id="scatter" viewBox="0 0 1100 540"></svg>
  <p class="foot">Circles are GRU, squares MLP, diamonds RCNN. Filled markers have three
  seeds; hollow ones are single-seed probes. The dashed line of each colour is that
  distance's MWPM.</p>
</div>

<div class="card">
  <h2>Validation loss — one line per seed, per configuration</h2>
  <svg id="curves" viewBox="0 0 1100 520"></svg>
  <p class="foot">Solid lines are GRU, dashed are MLP, dotted purple is the RCNN (which
  used early stopping, so it ends at epoch 34–42 rather than 200); colour is distance. Several lines
  share a colour because a distance has several configurations and three seeds each. The
  marker shows the epoch whose checkpoint was scored —
  d=5 peaks around epoch 30 and then overfits, while d=7 and d=9 keep improving to the cap.</p>
</div>

<div class="card">
  <h2>All runs</h2>
  <table id="tbl"></table>
  <p class="foot">Click a column heading to sort. The highlighted row is the only
  configuration that beat its MWPM.</p>
</div>
</div>
<div class="tt" id="tt"></div>
<script>
const DATA = __PAYLOAD__;
const C = {5:'#eb6834', 7:'#2a78d6', 9:'#1baf7a'};
const C_RCNN = '#8a5fb0';
const SHAPE = {gru:'circle', mlp:'square', rcnn:'diamond'};
const tt = document.getElementById('tt');
const state = {d:new Set([5,7,9]), arch:new Set(['gru','mlp','rcnn'])};

function show(e, html) {
  tt.innerHTML = html; tt.style.opacity = 1;
  tt.style.left = Math.min(e.clientX + 14, innerWidth - 260) + 'px';
  tt.style.top = (e.clientY + 14) + 'px';
}
function hide() { tt.style.opacity = 0; }
const el = (n, a) => { const x = document.createElementNS('http://www.w3.org/2000/svg', n);
  for (const k in a) x.setAttribute(k, a[k]); return x; };

function marker(shape, x, y, r, fill, stroke) {
  if (shape === 'circle') return el('circle', {cx:x, cy:y, r:r, fill:fill, stroke:stroke, 'stroke-width':2});
  if (shape === 'square') return el('rect', {x:x-r, y:y-r, width:2*r, height:2*r, fill:fill, stroke:stroke, 'stroke-width':2});
  return el('polygon', {points:`${x},${y-r-1} ${x+r+1},${y} ${x},${y+r+1} ${x-r-1},${y}`,
                        fill:fill, stroke:stroke, 'stroke-width':2});
}

function controls() {
  const box = document.getElementById('f1');
  const mk = (label, key, set) => {
    const b = document.createElement('button');
    b.textContent = label; b.dataset.k = key; b.className = 'on';
    b.onclick = () => { set.has(key) ? set.delete(key) : set.add(key);
      b.classList.toggle('on'); draw(); };
    box.appendChild(b);
  };
  [5,7,9].forEach(d => mk('d = ' + d, d, state.d));
  const gap = document.createElement('span'); gap.style.width = '18px'; box.appendChild(gap);
  [['GRU','gru'],['MLP','mlp'],['RCNN','rcnn']].forEach(([l,k]) => mk(l, k, state.arch));
}

function draw() {
  const svg = document.getElementById('scatter');
  svg.innerHTML = '';
  const W = 1100, H = 540, L = 74, R = 40, T = 18, B = 52;
  const xs = v => L + (Math.log10(v) - Math.log10(1.6e6)) / (Math.log10(34e6) - Math.log10(1.6e6)) * (W - L - R);
  const ys = v => T + (Math.log10(0.09) - Math.log10(v)) / (Math.log10(0.09) - Math.log10(0.0018)) * (H - T - B);

  [0.002,0.003,0.005,0.008,0.0125,0.02,0.03,0.05,0.08].forEach(v => {
    svg.appendChild(el('line', {x1:L, x2:W-R, y1:ys(v), y2:ys(v), stroke:'#e6e4e0'}));
    const t = el('text', {x:L-10, y:ys(v)+4, 'text-anchor':'end', fill:'#55534e', 'font-size':11});
    t.textContent = v; svg.appendChild(t);
  });
  [2e6,5e6,10e6,20e6].forEach(v => {
    const t = el('text', {x:xs(v), y:H-B+22, 'text-anchor':'middle', fill:'#55534e', 'font-size':11});
    t.textContent = (v/1e6) + 'M'; svg.appendChild(t);
    svg.appendChild(el('line', {x1:xs(v), x2:xs(v), y1:T, y2:H-B, stroke:'#f0eeeb'}));
  });
  let lab = el('text', {x:(L+W-R)/2, y:H-8, 'text-anchor':'middle', fill:'#1a1a1a', 'font-size':13});
  lab.textContent = 'training shots'; svg.appendChild(lab);
  lab = el('text', {x:16, y:(T+H-B)/2, fill:'#1a1a1a', 'font-size':13,
                    transform:`rotate(-90 16 ${(T+H-B)/2})`, 'text-anchor':'middle'});
  lab.textContent = 'logical error rate  p_L'; svg.appendChild(lab);

  for (const d of [5,7,9]) {
    if (!state.d.has(d)) continue;
    const m = DATA.mwpm[d];
    svg.appendChild(el('line', {x1:L, x2:W-R, y1:ys(m), y2:ys(m), stroke:C[d],
                                'stroke-dasharray':'6 4', 'stroke-width':1.5, opacity:0.6}));
    const t = el('text', {x:L+6, y:ys(m)-6, fill:C[d], 'font-size':11});
    t.textContent = 'MWPM d=' + d; svg.appendChild(t);
  }

  for (const d of [5,7,9]) {
    if (!state.d.has(d) || !state.arch.has('gru')) continue;
    const lad = DATA.runs.filter(r => r.d === d && r.label === 'GRU u140')
                         .sort((a,b) => a.shots - b.shots);
    if (lad.length < 2) continue;
    const pts = lad.map(r => `${xs(r.shots)},${ys(r.p_L)}`).join(' ');
    svg.appendChild(el('polyline', {points:pts, fill:'none', stroke:C[d], 'stroke-width':2.5}));
  }

  DATA.runs.forEach(r => {
    if (!state.d.has(r.d) || !state.arch.has(r.arch)) return;
    const g = marker(SHAPE[r.arch], xs(r.shots), ys(r.p_L), 7,
                     r.seeds === 3 ? C[r.d] : '#fff', C[r.d]);
    g.style.cursor = 'pointer';
    const ratio = (r.p_L / r.mwpm).toFixed(3);
    g.onmousemove = e => show(e,
      `<b>d=${r.d} · ${r.label}</b><br>${(r.shots/1e6)} M shots · ${r.seeds} seed${r.seeds>1?'s':''}<br>` +
      `p_L ${r.p_L.toFixed(6)}${r.sd ? ' ± ' + r.sd.toFixed(6) : ''}<br>` +
      `<b>${ratio}× MWPM</b> (${r.mwpm})<br>${r.params.toLocaleString()} params · ${r.minutes} min · ${r.machine}`);
    g.onmouseleave = hide;
    svg.appendChild(g);
  });

  const s = DATA.sealed;
  if (state.d.has(5) && state.arch.has('gru')) {
    const star = el('polygon', {fill:C[5], stroke:'#fff', 'stroke-width':1.5,
      points:Array.from({length:10}, (_, i) => {
        const a = -Math.PI/2 + i * Math.PI/5, rr = i % 2 ? 4.5 : 11;
        return `${xs(s.shots) + rr*Math.cos(a)},${ys(s.p_L) + rr*Math.sin(a)}`;
      }).join(' ')});
    star.onmousemove = e => show(e, `<b>sealed block ${s.block}</b><br>d=5 GRU u140, 20M augmented<br>` +
      `p_L ${s.p_L.toFixed(6)} ± ${s.sd.toFixed(6)}<br><b>${(s.p_L/s.mwpm).toFixed(3)}× MWPM</b> (${s.mwpm})`);
    star.onmouseleave = hide;
    svg.appendChild(star);
  }
}

function drawCurves() {
  const svg = document.getElementById('curves');
  const W = 1100, H = 520, L = 74, R = 30, T = 14, B = 48;
  const xs = e => L + e / 200 * (W - L - R);
  const ys = v => T + (Math.log10(0.42) - Math.log10(v)) / (Math.log10(0.42) - Math.log10(0.018)) * (H - T - B);
  [0.02,0.03,0.05,0.08,0.12,0.2,0.3,0.4].forEach(v => {
    svg.appendChild(el('line', {x1:L, x2:W-R, y1:ys(v), y2:ys(v), stroke:'#e6e4e0'}));
    const t = el('text', {x:L-10, y:ys(v)+4, 'text-anchor':'end', fill:'#55534e', 'font-size':11});
    t.textContent = v; svg.appendChild(t);
  });
  [0,50,100,150,200].forEach(e => {
    const t = el('text', {x:xs(e), y:H-B+22, 'text-anchor':'middle', fill:'#55534e', 'font-size':11});
    t.textContent = e; svg.appendChild(t);
  });
  let lab = el('text', {x:(L+W-R)/2, y:H-8, 'text-anchor':'middle', fill:'#1a1a1a', 'font-size':13});
  lab.textContent = 'epoch'; svg.appendChild(lab);

  DATA.curves.forEach(c => {
    const step = 2, pts = [];
    for (let i = 0; i < c.val.length; i += step) pts.push(`${xs(i+1)},${ys(c.val[i])}`);
    const col = c.arch === 'rcnn' ? C_RCNN : C[c.d];
    const dash = {gru:'', mlp:'5 3', rcnn:'2 3'}[c.arch];
    const path = el('polyline', {points:pts.join(' '), fill:'none', stroke:col,
                                 'stroke-width':c.arch === 'rcnn' ? 2.4 : 1.6,
                                 opacity:c.arch === 'rcnn' ? 0.95 : (c.arch === 'gru' ? 0.8 : 0.5),
                                 'stroke-dasharray':dash});
    path.style.cursor = 'pointer';
    path.onmousemove = ev => show(ev, `<b>d=${c.d} · ${c.arch.toUpperCase()}${c.units ? ' u' + c.units : ''}</b><br>` +
      `${c.shots/1e6} M shots · seed ${c.seed}<br>best epoch ${c.best+1} · val ${c.val[c.best]}`);
    path.onmouseleave = hide;
    svg.appendChild(path);
    svg.appendChild(el('circle', {cx:xs(c.best+1), cy:ys(c.val[c.best]), r:3.8, fill:col,
                                  stroke:'#fff', 'stroke-width':1.2}));
  });
}

function table() {
  const cols = [['d','d'],['label','model'],['shots','shots'],['seeds','seeds'],
                ['p_L','p_L'],['mwpm','MWPM'],['ratio','×MWPM'],['params','params'],
                ['minutes','minutes'],['machine','machine'],['date','date']];
  const rows = DATA.runs.map(r => ({...r, ratio:r.p_L / r.mwpm}));
  rows.push({...DATA.sealed, arch:'gru', label:'GRU u140 20M — SEALED', batch:10000,
             ratio:DATA.sealed.p_L / DATA.sealed.mwpm, params:69441, minutes:55,
             machine:'GH200', date:'08-21'});
  let dir = 1, key = 'ratio';
  const render = () => {
    const t = document.getElementById('tbl');
    rows.sort((a,b) => (a[key] > b[key] ? 1 : -1) * dir);
    t.innerHTML = '<tr>' + cols.map(c => `<th data-k="${c[0]}">${c[1]}</th>`).join('') + '</tr>' +
      rows.map(r => `<tr class="${r.ratio < 1 ? 'win' : ''}">` +
        `<td><span class="dot" style="background:${C[r.d]}"></span>${r.d}</td>` +
        `<td>${r.label}</td><td>${r.shots/1e6}M</td><td>${r.seeds}</td>` +
        `<td>${r.p_L.toFixed(6)}${r.sd ? ' ± ' + r.sd.toFixed(6) : ''}</td>` +
        `<td>${r.mwpm}</td><td><b>${r.ratio.toFixed(3)}×</b></td>` +
        `<td>${r.params.toLocaleString()}</td><td>${r.minutes}</td>` +
        `<td>${r.machine}</td><td>${r.date}</td></tr>`).join('');
    t.querySelectorAll('th').forEach(th => th.onclick = () => {
      const k = th.dataset.k; dir = (k === key) ? -dir : 1; key = k; render();
    });
  };
  render();
}

controls(); draw(); drawCurves(); table();
</script>
"""
html = html.replace('__PAYLOAD__', payload)
os.makedirs(OUT, exist_ok=True)
path = os.path.join(OUT, 'exp16_results.html')
open(path, 'w').write(html)
print(f'wrote {path}  ({len(html)/1024:.0f} KB, {len(curves)} curves, {len(rows())} runs)')
