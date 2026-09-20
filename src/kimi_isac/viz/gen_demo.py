"""Generative reconstruction demo: echo -> cloud vs class prior.

Loads a trained checkpoint (default results/gen/seed_1001 from the full
report), reconstructs a few stratified evaluation samples, and writes a
self-contained HTML comparing ground truth, conditional reconstruction,
and the unconditional class-prior output.

Usage: python -m kimi_isac.viz.gen_demo [--checkpoint-dir results/gen/seed_1001]
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import torch

from kimi_isac.core import logging_setup
from kimi_isac.gen import dataset as ds
from kimi_isac.gen import dit, metrics
from kimi_isac.gen import vae as vae_mod
from kimi_isac.ml.checkpoint import load_checkpoint

N_SAMPLES = 4


def load_model(checkpoint_dir: Path, cfg: ds.GenConfig, tau: int):
    vae = vae_mod.PointVAE()
    load_checkpoint(checkpoint_dir / "vae" / "best.pth", vae)
    vae.eval()
    enc = dit.CondEncoder(n_classes=10, feat_dim=10, tau=tau)
    load_checkpoint(checkpoint_dir / "dit" / "enc_best.pth", enc)
    enc.eval()
    model = dit.LatentDiT()
    load_checkpoint(checkpoint_dir / "dit" / "best.pth", model)
    model.eval()
    return vae, enc, model


@torch.no_grad()
def build_payload(checkpoint_dir: Path, cfg: ds.GenConfig, T: int = 100) -> dict:
    from kimi_isac.gen import echo, templates
    from kimi_isac.gen.dataset import CELL_EVAL_BASE

    vae, enc, model = load_model(checkpoint_dir, cfg, cfg.tau)
    sched = dit.DDPMScheduler(T=T)

    # four samples spanning (snr, mode, class) — same grid the report uses
    picks = {
        (6.0, "aligned", "uav"),
        (1.0, "aligned", "building"),
        (6.0, "none", "vehicle"),
        (0.25, "random", "tower"),
    }
    items = []
    counter = 0
    for snr in cfg.snr_levels:
        for mode in ("aligned", "random", "none"):
            for cls_name in templates.TRAIN_CLASSES:
                for k in range(2):
                    idx = CELL_EVAL_BASE + counter
                    counter += 1
                    if (snr, mode, cls_name) not in picks or k != 0:
                        continue
                    rng = np.random.default_rng((cfg.seed * 1_000_003 + idx) % (2**63))
                    cloud = templates.make_template(cls_name, rng)
                    out = echo.echo_and_features(
                        cloud,
                        cls_name,
                        mode,
                        seed=idx,
                        snr=float(snr),
                        cfg=echo.EchoConfig(n_elem=cfg.n_elem, tau=cfg.tau),
                    )
                    features = out["features"].astype(np.float32)
                    cls = templates.class_index(cls_name)
                    feats_t = torch.from_numpy(features).unsqueeze(0)
                    pooled, tokens = enc(torch.tensor([cls]), feats_t)
                    z = sched.sample(model, pooled, tokens, shape=(1, 256))
                    cond_cloud = vae.decode(z).numpy()[0]
                    zeros = torch.zeros_like(feats_t)
                    pooled0, tokens0 = enc(torch.tensor([cls]), zeros)
                    z0 = sched.sample(model, pooled0, tokens0, shape=(1, 256))
                    prior_cloud = vae.decode(z0).numpy()[0]
                    items.append(
                        {
                            "label": f"{cls_name} · SNR {snr:g} · RIS {mode}",
                            "features": features.tolist(),
                            "gt": cloud.astype(np.float32).tolist(),
                            "cond": cond_cloud.astype(np.float32).tolist(),
                            "prior": prior_cloud.astype(np.float32).tolist(),
                            "cd_cond": metrics.chamfer_distance(cond_cloud, cloud),
                            "cd_prior": metrics.chamfer_distance(prior_cloud, cloud),
                        }
                    )
    return {"items": items}


_HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>kimi-isac · echo-conditioned 3D reconstruction vs class prior</title>
<style>
 body { font-family: system-ui, sans-serif; margin: 24px; background: #fafafa; }
 .wrap { max-width: 1100px; margin: 0 auto; }
 .sample { background: #fff; border: 1px solid #ddd; border-radius: 8px;
           padding: 14px; margin-bottom: 16px; }
 h1 { font-size: 20px; } h2 { font-size: 15px; margin: 2px 0 10px; }
 .row { display: flex; gap: 14px; flex-wrap: wrap; align-items: flex-start; }
 canvas { background: #fff; border: 1px solid #eee; border-radius: 6px; }
 .tag { font-size: 13px; padding: 2px 8px; border-radius: 10px; color: #fff; }
 .clouds { display: flex; gap: 10px; }
 .cell { text-align: center; }
 .feat { border-collapse: collapse; font-size: 10px; }
 .feat td { width: 26px; height: 14px; text-align: center; color: #fff; }
 input[type=range] { width: 140px; }
</style>
</head>
<body>
<div class="wrap">
<h1>Echo-conditioned reconstruction vs class prior (PointVAE + latent DiT)</h1>
<p style="font-size:13px;color:#555">Drag to rotate. Blue: ground truth. Orange: conditional
reconstruction from the echo features. Grey: unconditional class-prior sample. The CD badges
report Chamfer distance; on single-station echoes the prior is a strong baseline (see TECH_REPORT §7).</p>
<div id="samples"></div>
</div>
<script>
const DATA = __PAYLOAD__;
const FEAT_NAMES = ['amp_dB','sin','cos','doppler','delay_ms','dist','elev','rcs','irs_sin','irs_cos'];

function heatColor(v, lo, hi) {
  const t = Math.max(0, Math.min(1, (v - lo) / (hi - lo + 1e-9)));
  const c = Math.round(255 * (0.2 + 0.8 * t));
  return `rgb(${c},${Math.round(120 + 100 * t)},${Math.round(255 - 150 * t)})`;
}

function rotProject(p, yaw, pitch) {
  const [x, y, z] = p;
  const cy = Math.cos(yaw), sy = Math.sin(yaw);
  const x1 = cy * x - sy * y, y1 = sy * x + cy * y;
  const cp = Math.cos(pitch), sp = Math.sin(pitch);
  const z1 = cp * z - sp * y1, y2 = sp * z + cp * y1;
  return [x1, y2, z1];
}

function drawCloud(canvas, cloud, color, yaw, pitch) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const pts = cloud.map(p => rotProject(p, yaw, pitch));
  const xs = pts.map(p => p[0]), ys = pts.map(p => p[2]);
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const minY = Math.min(...ys), maxY = Math.max(...ys);
  const sx = (canvas.width - 20) / (maxX - minX + 1e-9);
  const sy = (canvas.height - 20) / (maxY - minY + 1e-9);
  const s = Math.min(sx, sy);
  ctx.fillStyle = color;
  pts.forEach(p => {
    const px = 10 + (p[0] - minX) * s;
    const py = canvas.height - 10 - (p[2] - minY) * s;
    ctx.fillRect(px, py, 1.4, 1.4);
  });
}

const host = document.getElementById('samples');
DATA.items.forEach((item, idx) => {
  const div = document.createElement('div');
  div.className = 'sample';
  const yawId = 'yaw' + idx;
  div.innerHTML = `
    <h2>${item.label}
      <span class="tag" style="background:#1f77b4">conditional CD ${item.cd_cond.toFixed(4)}</span>
      <span class="tag" style="background:#999">prior CD ${item.cd_prior.toFixed(4)}</span>
    </h2>
    <div class="row">
      <div class="clouds">
        <div class="cell"><canvas id="gt${idx}" width="220" height="220"></canvas><div>ground truth</div></div>
        <div class="cell"><canvas id="cond${idx}" width="220" height="220"></canvas><div>conditional (echo)</div></div>
        <div class="cell"><canvas id="prior${idx}" width="220" height="220"></canvas><div>prior (class only)</div></div>
      </div>
      <div>
        <div style="font-size:12px;margin-bottom:4px">echo features (frames × 10)</div>
        <table class="feat" id="feat${idx}"></table>
        <div style="margin-top:6px"><input type="range" id="${yawId}" min="-3.14" max="3.14" step="0.05" value="0.6"> rotate</div>
      </div>
    </div>`;
  host.appendChild(div);

  // feature heatmap
  const tbl = div.querySelector('#feat' + idx);
  const flat = item.features.flat();
  const lo = Math.min(...flat), hi = Math.max(...flat);
  item.features.forEach((row, r) => {
    const tr = document.createElement('tr');
    row.forEach((v, c) => {
      const td = document.createElement('td');
      td.style.background = heatColor(v, lo, hi);
      td.textContent = v.toFixed(1);
      td.title = FEAT_NAMES[c];
      tr.appendChild(td);
    });
    tbl.appendChild(tr);
  });

  const render = () => {
    const yaw = +div.querySelector('#' + yawId).value;
    drawCloud(div.querySelector('#gt' + idx), item.gt, '#1f77b4', yaw, 0.5);
    drawCloud(div.querySelector('#cond' + idx), item.cond, '#e08214', yaw, 0.5);
    drawCloud(div.querySelector('#prior' + idx), item.prior, '#999', yaw, 0.5);
  };
  div.querySelector('#' + yawId).addEventListener('input', render);
  render();
});
</script>
</body>
</html>
"""


def write_html(payload: dict, path: Path) -> Path:
    body = json.dumps(payload, separators=(",", ":"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(_HTML_TEMPLATE.replace("__PAYLOAD__", body), encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint-dir", type=str, default="results/gen/seed_1001")
    parser.add_argument("--out", type=str, default="docs/demos")
    args = parser.parse_args(argv)

    logging_setup.setup_logging()
    log = logging.getLogger("kimi_isac.viz.gen_demo")
    ckpt = Path(args.checkpoint_dir)
    if not (ckpt / "dit" / "best.pth").exists():
        log.error("no checkpoint at %s; run the full gen report first", ckpt)
        return 1
    cfg = ds.GenConfig()
    payload = build_payload(ckpt, cfg)
    path = write_html(payload, Path(args.out) / "gen_reconstruction.html")
    log.info("wrote %s (%d KB)", path, path.stat().st_size // 1024)
    for it in payload["items"]:
        log.info("%-28s cond CD %.4f | prior CD %.4f", it["label"], it["cd_cond"], it["cd_prior"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
