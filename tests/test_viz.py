"""Tests for demo asset generation (self-contained HTML + valid GIF)."""

import json

from kimi_isac.viz import closedloop_demo, gen_demo


def test_collect_scenario_shapes():
    data = closedloop_demo.collect(n_frames=8, dt_s=0.5, n_elem=256)
    assert data["meta"]["n_frames"] == 8
    assert len(data["snr_db"]["tracked"]) == 8
    assert len(data["sat_xy_km"]) == 8
    assert len(data["doppler_hz"]) == 8
    assert data["snr_db"]["tracked"][0] > data["snr_db"]["random"][0] + 5.0


def test_html_is_self_contained(tmp_path):
    data = closedloop_demo.collect(n_frames=4, dt_s=0.5, n_elem=256)
    path = closedloop_demo.write_html(data, tmp_path / "demo.html")
    text = path.read_text(encoding="utf-8")
    assert "const DATA = {" in text
    assert "http://" not in text and "https://" not in text  # no CDN
    payload = json.loads(text.split("const DATA = ", 1)[1].split(";\n", 1)[0])
    assert payload["meta"]["n_frames"] == 4


def test_gif_is_valid(tmp_path):
    data = closedloop_demo.collect(n_frames=4, dt_s=0.5, n_elem=256)
    path = closedloop_demo.write_gif(data, tmp_path / "demo.gif")
    assert path.read_bytes()[:6] in (b"GIF87a", b"GIF89a")


def test_gen_demo_html_structure(tmp_path):
    from kimi_isac.gen import dataset as ds
    from kimi_isac.gen import train as gen_train

    cfg = ds.GenConfig(n_total=24, seed=0, snr_levels=(6.0, 1.0, 0.25), tau=4, n_elem=16)
    gen_train.run_gen(
        seed=0,
        cfg=cfg,
        epochs_vae=1,
        epochs_dit=1,
        device="cpu",
        out_dir=tmp_path,
        log=__import__("logging").getLogger("test"),
        T=20,
    )
    payload = gen_demo.build_payload(tmp_path, cfg, T=20)
    assert len(payload["items"]) == 4
    for it in payload["items"]:
        assert len(it["features"]) == cfg.tau
        assert len(it["gt"]) == 512
        assert it["cd_cond"] >= 0.0 and it["cd_prior"] >= 0.0
    path = gen_demo.write_html(payload, tmp_path / "gen.html")
    text = path.read_text(encoding="utf-8")
    assert "const DATA = {" in text
    assert "http://" not in text and "https://" not in text
