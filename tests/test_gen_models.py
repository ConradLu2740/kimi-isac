"""Tests for the generative models."""

import torch

from kimi_isac.gen import dit, vae


def test_vae_roundtrip_shapes():
    m = vae.PointVAE(n_points=512, z_dim=256)
    x = torch.randn(4, 512, 3)
    out = m(x)
    assert out["x_hat"].shape == x.shape
    mu, logvar = m.encode(x)
    assert mu.shape == (4, 256)
    loss = vae.vae_loss(out, x, kl_weight=1e-4)
    assert loss.dim() == 0


def test_cond_encoder_shapes():
    enc = dit.CondEncoder(n_classes=10, feat_dim=10, tau=8)
    cls = torch.zeros(4, dtype=torch.long)
    feats = torch.randn(4, 8, 10)
    pooled, tokens = enc(cls, feats)
    assert pooled.shape == (4, 256)
    assert tokens.shape == (4, 8, 128)


def test_dit_forward_and_scheduler():
    model = dit.LatentDiT(z_dim=256, n_tokens=8)
    sched = dit.DDPMScheduler(T=100)
    enc = dit.CondEncoder(n_classes=10, feat_dim=10, tau=8)
    cls = torch.zeros(2, dtype=torch.long)
    feats = torch.randn(2, 8, 10)
    pooled, tokens = enc(cls, feats)
    z = torch.randn(2, 256)
    t = torch.tensor([10, 50])
    noisy = sched.add_noise(z, t, torch.randn_like(z))
    assert noisy.shape == z.shape
    eps = model(noisy, t, pooled, tokens)
    assert eps.shape == z.shape


def test_scheduler_sampling_shapes():
    model = dit.LatentDiT(z_dim=256, n_tokens=8)
    sched = dit.DDPMScheduler(T=10)
    enc = dit.CondEncoder(n_classes=10, feat_dim=10, tau=8)
    pooled, tokens = enc(torch.zeros(2, dtype=torch.long), torch.randn(2, 8, 10))
    z = sched.sample(model, pooled, tokens, shape=(2, 256))
    assert z.shape == (2, 256)
