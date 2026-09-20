"""Latent DiT denoiser with cross-attention conditioning + linear-beta DDPM."""

from __future__ import annotations

import torch
from torch import nn


class CondEncoder(nn.Module):
    """Class embedding + feature-sequence Transformer -> pooled conditioning."""

    def __init__(
        self,
        n_classes: int = 10,
        feat_dim: int = 10,
        tau: int = 8,
        d_model: int = 128,
        n_heads: int = 4,
        out_dim: int = 256,
    ):
        super().__init__()
        self.class_emb = nn.Embedding(n_classes, d_model)
        self.feat_proj = nn.Linear(feat_dim, d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=4 * d_model, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=2)
        self.pool = nn.Linear(d_model, out_dim)

    def forward(self, class_idx: torch.Tensor, features: torch.Tensor):
        tokens = self.feat_proj(features) + self.class_emb(class_idx).unsqueeze(1)
        tokens = self.transformer(tokens)
        return self.pool(tokens.mean(dim=1)), tokens


class DiTBlock(nn.Module):
    """AdaLN-zero self-attention + cross-attention to condition tokens."""

    def __init__(self, hidden: int, n_heads: int, cond_dim: int, cond_token_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden, elementwise_affine=False)
        self.cross = nn.MultiheadAttention(
            hidden, n_heads, batch_first=True, kdim=cond_token_dim, vdim=cond_token_dim
        )
        self.norm3 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, 4 * hidden), nn.GELU(), nn.Linear(4 * hidden, hidden)
        )
        self.ada = nn.Sequential(nn.SiLU(), nn.Linear(cond_dim, 6 * hidden))
        ada_out = self.ada[1]
        assert isinstance(ada_out, nn.Linear)
        nn.init.zeros_(ada_out.weight)
        nn.init.zeros_(ada_out.bias)

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor, cond_tokens: torch.Tensor
    ) -> torch.Tensor:
        shift1, scale1, gate1, shift2, scale2, gate2 = self.ada(cond).chunk(6, dim=-1)
        h = self.norm1(x) * (1 + scale1.unsqueeze(1)) + shift1.unsqueeze(1)
        x = x + gate1.unsqueeze(1) * self.attn(h, h, h)[0]
        h2 = self.norm2(x) * (1 + scale2.unsqueeze(1)) + shift2.unsqueeze(1)
        x = x + self.cross(h2, cond_tokens, cond_tokens)[0]
        return x + self.mlp(self.norm3(x))


class LatentDiT(nn.Module):
    def __init__(
        self,
        z_dim: int = 256,
        n_tokens: int = 8,
        depth: int = 4,
        hidden: int = 256,
        n_heads: int = 8,
        cond_dim: int = 256,
        cond_token_dim: int = 128,
    ):
        super().__init__()
        self.n_tokens = n_tokens
        self.in_proj = nn.Linear(z_dim // n_tokens, hidden)
        self.pos = nn.Parameter(torch.randn(1, n_tokens, hidden) * 0.02)
        self.blocks = nn.ModuleList(
            [
                DiTBlock(hidden, n_heads, cond_dim=cond_dim, cond_token_dim=cond_token_dim)
                for _ in range(depth)
            ]
        )
        self.out_norm = nn.LayerNorm(hidden)
        self.out_proj = nn.Linear(hidden, z_dim // n_tokens)

    def forward(
        self,
        z_noisy: torch.Tensor,
        t: torch.Tensor,
        cond_emb: torch.Tensor,
        cond_tokens: torch.Tensor,
    ) -> torch.Tensor:
        h = z_noisy.reshape(z_noisy.shape[0], self.n_tokens, -1)
        h = self.in_proj(h) + self.pos
        for blk in self.blocks:
            h = blk(h, cond_emb, cond_tokens)
        return self.out_proj(self.out_norm(h)).reshape(z_noisy.shape[0], -1)


class DDPMScheduler:
    """Linear-beta DDPM, T = 100, deterministic ancestral sampling."""

    def __init__(self, T: int = 100, beta_start: float = 1e-4, beta_end: float = 0.02):
        self.T = T
        self.betas = torch.linspace(beta_start, beta_end, T)
        self.alphas_cumprod = torch.cumprod(1.0 - self.betas, dim=0)

    def add_noise(self, z: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        ac = self.alphas_cumprod.to(z.device)[t].view(-1, 1)
        return ac.sqrt() * z + (1 - ac).sqrt() * noise

    @torch.no_grad()
    def sample(
        self,
        model: nn.Module,
        cond_emb: torch.Tensor,
        cond_tokens: torch.Tensor,
        shape: tuple[int, ...],
        device: str = "cpu",
    ) -> torch.Tensor:
        """Ancestral sampling: z_T -> z_0 through the learned eps network."""
        z = torch.randn(shape, device=device)
        for step in reversed(range(self.T)):
            t = torch.full((shape[0],), step, device=device, dtype=torch.long)
            eps = model(z, t, cond_emb, cond_tokens)
            beta = self.betas.to(device)[step]
            alpha = self.alphas_cumprod.to(device)[step]
            mean = (z - beta / (1 - alpha).sqrt() * eps) / (1 - beta).sqrt()
            if step > 0:
                z = mean + beta.sqrt() * torch.randn_like(z)
            else:
                z = mean
        return z
