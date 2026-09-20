"""Closed-loop demo: ISS overpass with RIS phase control.

Runs the deterministic scenario (fixed seed), then writes:
- docs/demos/closedloop_demo.html  (interactive: frame slider, SNR curves,
  geometry projection, Doppler readout)
- docs/demos/closedloop.gif        (animation of the same data)
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from kimi_isac.closedloop import frame_paths, iss_overpass_geometry
from kimi_isac.core import logging_setup
from kimi_isac.gen.scene import ue_and_frame
from kimi_isac.opt.phase_opt import align_phases, segment_track

VARIANTS = ("direct_only", "random", "tracked", "segmented_16", "segmented_8", "segmented_4")


def _project(
    sat_ecef: np.ndarray, ue: np.ndarray, up: np.ndarray, east: np.ndarray
) -> tuple[float, float]:
    """Project an ECEF point to the local (east, north) plane at the UE, in km."""
    v = np.asarray(sat_ecef, dtype=float) - np.asarray(ue, dtype=float)
    north = np.cross(up, east)
    return float(np.dot(v, east)) / 1e3, float(np.dot(v, north)) / 1e3


def collect(
    n_frames: int = 24,
    dt_s: float = 0.5,
    n_elem: int = 16384,
    blockage_db: float = 60.0,
    seed: int = 42,
) -> dict:
    """Run the scenario once and gather per-frame data for rendering."""
    from kimi_isac.closedloop import snr_db

    rng = np.random.default_rng(seed)
    geometry = iss_overpass_geometry(n_frames=n_frames, dt_s=dt_s, n_elem=n_elem)
    ue, up, east = ue_and_frame()
    paths = [frame_paths(g, blockage_db) for g in geometry]
    tracked = np.array(
        [align_phases(a, h_d, seed=int(rng.integers(1 << 31))) for a, h_d, _ in paths]
    )
    random_phi = rng.uniform(0.0, 2.0 * np.pi, size=(n_frames, n_elem))
    variants = {
        "direct_only": np.zeros((n_frames, n_elem)),
        "random": random_phi,
        "tracked": tracked,
        "segmented_16": segment_track(tracked, 16),
        "segmented_8": segment_track(tracked, 8),
        "segmented_4": segment_track(tracked, 4),
    }
    snr = {
        name: [snr_db(a, h_d, phi[k]) for k, (a, h_d, _) in enumerate(paths)]
        for name, phi in variants.items()
    }
    sat_xy = [_project(g["sat_pos"], ue, up, east) for g in geometry]
    ris_xy = [_project(g["ris"][g["ris"].shape[0] // 2], ue, up, east) for g in geometry]
    doppler = [fd for _, _, fd in paths]
    elevation = []
    for g in geometry:
        v = np.asarray(g["sat_pos"], dtype=float) - ue
        sin_el = float(np.dot(v, up) / (np.linalg.norm(v) + 1e-12))
        elevation.append(float(np.degrees(np.arcsin(np.clip(sin_el, -1.0, 1.0)))))
    return {
        "meta": {
            "n_frames": n_frames,
            "dt_s": dt_s,
            "n_elem": n_elem,
            "blockage_db": blockage_db,
            "carrier_hz": 30.0e9,
        },
        "variants": list(variants),
        "snr_db": snr,
        "sat_xy_km": sat_xy,
        "ris_xy_km": ris_xy,
        "doppler_hz": doppler,
        "elevation_deg": elevation,
    }


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>kimi-isac · RIS closed loop over an ISS overpass</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; }
 .wrap { max-width: 980px; margin: 0 auto; }
 canvas { background: #fff; border: 1px solid #ddd; border-radius: 6px; }
 .row { display: flex; gap: 16px; flex-wrap: wrap; }
 .controls { margin: 12px 0; display: flex; gap: 12px; align-items: center; }
 input[type=range] { width: 420px; }
 .readout { font-variant-numeric: tabular-nums; font-size: 14px; }
 h1 { font-size: 20px; } h2 { font-size: 15px; margin: 8px 0 4px; }
 .legend span { display: inline-block; margin-right: 14px; font-size: 13px; }
 .sw { display: inline-block; width: 12px; height: 12px; border-radius: 2px; vertical-align: -1px; }
</style>
</head>
<body>
<div class="wrap">
<h1>RIS closed loop over a real ISS overpass (30 GHz, 60 dB blockage, 16 384-element panel)</h1>
<div class="controls">
  <button id="play">&#9654; play</button>
  <input id="slider" type="range" min="0" max="0" value="0" step="1">
  <span class="readout" id="frameLabel"></span>
</div>
<div class="row">
  <canvas id="geo" width="460" height="380"></canvas>
  <canvas id="snr" width="460" height="380"></canvas>
</div>
<div class="legend" id="legend"></div>
<div class="readout" id="metrics"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const COLORS = {direct_only: '#888', random: '#e08214', tracked: '#1f77b4',
                segmented_16: '#2ca02c', segmented_8: '#d62728', segmented_4: '#9467bd'};
let frame = 0, timer = null;
const slider = document.getElementById('slider');
slider.max = DATA.meta.n_frames - 1;
const legend = document.getElementById('legend');
legend.innerHTML = DATA.variants.map(v =>
  `<span><span class="sw" style="background:${COLORS[v]}"></span> ${v}</span>`).join('');

function mean(a){ return a.reduce((x,y)=>x+y,0)/a.length; }

function drawGeo() {
  const c = document.getElementById('geo'), x = c.getContext('2d');
  x.clearRect(0,0,c.width,c.height);
  const s = DATA.sat_xy_km, r = DATA.ris_xy_km;
  const all = s.concat([r[0]]);
  const xs = all.map(p=>p[0]), ys = all.map(p=>p[1]);
  const minX = Math.min(...xs)-20, maxX = Math.max(...xs)+20;
  const minY = Math.min(...ys)-20, maxY = Math.max(...ys)+20;
  const P = (p) => [ (p[0]-minX)/(maxX-minX)*(c.width-40)+20,
                     c.height-40-(p[1]-minY)/(maxY-minY)*(c.height-40) ];
  x.strokeStyle = '#ccc'; x.lineWidth = 1;
  x.strokeRect(20, 20, c.width-40, c.height-40);
  // UE at origin
  const [ux, uy] = P([0,0]);
  x.fillStyle = '#000'; x.beginPath(); x.arc(ux, uy, 4, 0, 7); x.fill();
  x.fillText('UE', ux+6, uy+4);
  // RIS panel
  const [rx, ry] = P(r[frame]);
  x.fillStyle = '#9467bd'; x.fillRect(rx-6, ry-3, 12, 6);
  x.fillText('RIS', rx+8, ry+4);
  // satellite trajectory + current position
  x.strokeStyle = '#bbb'; x.beginPath();
  s.forEach((p,i)=>{ const [px,py]=P(p); i?x.lineTo(px,py):x.moveTo(px,py); }); x.stroke();
  const [sx, sy] = P(s[frame]);
  x.fillStyle = '#d62728'; x.beginPath(); x.arc(sx, sy, 6, 0, 7); x.fill();
  x.fillText('sat', sx+8, sy+4);
  x.fillStyle = '#333'; x.font = '12px system-ui';
  x.fillText('east (km)', c.width-90, c.height-10);
}

function drawSnr() {
  const c = document.getElementById('snr'), x = c.getContext('2d');
  x.clearRect(0,0,c.width,c.height);
  const vals = DATA.variants.flatMap(v => DATA.snr_db[v]);
  const lo = Math.min(...vals)-3, hi = Math.max(...vals)+3;
  const X = i => 40 + i/(DATA.meta.n_frames-1)*(c.width-60);
  const Y = v => c.height-30 - (v-lo)/(hi-lo)*(c.height-60);
  x.strokeStyle = '#999'; x.beginPath(); x.moveTo(40, Y(0)); x.lineTo(c.width-20, Y(0)); x.stroke();
  DATA.variants.forEach(v => {
    x.strokeStyle = COLORS[v]; x.lineWidth = v==='tracked' ? 2.4 : 1.4;
    x.beginPath();
    DATA.snr_db[v].forEach((s,i)=> i?x.lineTo(X(i),Y(s)):x.moveTo(X(i),Y(s)));
    x.stroke();
  });
  x.fillStyle = '#333';
  x.fillText('SNR (dB)', 6, 16);
  x.fillText('frame', c.width-60, c.height-10);
  // current-frame marker
  DATA.variants.forEach(v => {
    x.fillStyle = COLORS[v];
    x.beginPath(); x.arc(X(frame), Y(DATA.snr_db[v][frame]), 3.5, 0, 7); x.fill();
  });
}

function update() {
  document.getElementById('frameLabel').textContent =
    `frame ${frame+1}/${DATA.meta.n_frames} (t = ${(frame*DATA.meta.dt_s).toFixed(1)} s)`;
  slider.value = frame;
  drawGeo(); drawSnr();
  document.getElementById('metrics').textContent =
    `Doppler ${DATA.doppler_hz[frame].toFixed(0)} Hz · elevation ${DATA.elevation_deg[frame].toFixed(1)}° · ` +
    DATA.variants.map(v => `${v} ${DATA.snr_db[v][frame].toFixed(1)} dB`).join(' · ') +
    ` · mean gain tracked vs random ${(mean(DATA.snr_db.tracked)-mean(DATA.snr_db.random)).toFixed(1)} dB`;
}
slider.addEventListener('input', e => { frame = +e.target.value; update(); });
document.getElementById('play').addEventListener('click', function() {
  if (timer) { clearInterval(timer); timer = null; this.textContent = '\\u25B6 play'; return; }
  this.textContent = '\\u23F8 pause';
  timer = setInterval(() => { frame = (frame+1) % DATA.meta.n_frames; update(); }, 400);
});
update();
</script>
</body>
</html>
"""


def write_html(data: dict, path: Path) -> Path:
    payload = json.dumps(data, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HTML_TEMPLATE.replace("__PAYLOAD__", payload), encoding="utf-8")
    return path


def write_gif(data: dict, path: Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    n = data["meta"]["n_frames"]
    fig, (ax_geo, ax_snr) = plt.subplots(1, 2, figsize=(9, 3.6), dpi=80)
    traj = np.asarray(data["sat_xy_km"])
    ax_geo.plot(traj[:, 0], traj[:, 1], ".-", color="#bbb", ms=2, lw=0.8)
    ax_geo.plot([0], [0], "o", color="k", ms=4)
    ax_geo.plot(data["ris_xy_km"][0][0], data["ris_xy_km"][0][1], "s", color="#9467bd")
    ax_geo.set_xlabel("east (km)")
    ax_geo.set_ylabel("north (km)")
    (sat_dot,) = ax_geo.plot([], [], "o", color="tab:red", ms=7)
    lines = {}
    for v in data["variants"]:
        (lines[v],) = ax_snr.plot([], [], label=v, lw=1.6)
    ax_snr.set_xlabel("frame")
    ax_snr.set_ylabel("SNR (dB)")
    ax_snr.legend(fontsize=7, ncol=2)

    def update(frame: int):
        sat_dot.set_data([data["sat_xy_km"][frame][0]], [data["sat_xy_km"][frame][1]])
        ax_geo.relim()
        ax_geo.autoscale_view()
        xs = np.arange(frame + 1)
        for v in data["variants"]:
            lines[v].set_data(xs, data["snr_db"][v][: frame + 1])
        ax_snr.relim()
        ax_snr.autoscale_view()
        return list(lines.values()) + [sat_dot]

    anim = FuncAnimation(fig, update, frames=n, interval=400, blit=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    anim.save(path, writer=PillowWriter(fps=2))
    plt.close(fig)
    return path


def main(out_dir: str = "docs/demos") -> int:
    import logging

    logging_setup.setup_logging()
    log = logging.getLogger("kimi_isac.viz.closedloop")
    data = collect()
    out = Path(out_dir)
    html = write_html(data, out / "closedloop_demo.html")
    gif = write_gif(data, out / "closedloop.gif")
    log.info(
        "wrote %s (%d KB) and %s (%d KB)",
        html,
        html.stat().st_size // 1024,
        gif,
        gif.stat().st_size // 1024,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
