"""Point-cloud VAE: 512x3 -> z(256)."""

from __future__ import annotations

import torch
from torch import nn


class PointVAE(nn.Module):
    def __init__(self, n_points: int = 512, z_dim: int = 256):
        super().__init__()
        self.n_points = n_points
        self.z_dim = z_dim
        in_dim = n_points * 3
        self.encoder = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(512, z_dim)
        self.fc_logvar = nn.Linear(512, z_dim)
        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, in_dim),
        )

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x.reshape(x.shape[0], -1))
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        return mu + std * torch.randn_like(std)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z).reshape(-1, self.n_points, 3)

    def forward(self, x: torch.Tensor) -> dict:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return {"x_hat": self.decode(z), "mu": mu, "logvar": logvar, "z": z}


def vae_loss(out: dict, x: torch.Tensor, kl_weight: float = 1e-4) -> torch.Tensor:
    recon = ((out["x_hat"] - x) ** 2).sum(dim=-1).mean()
    mu, logvar = out["mu"], out["logvar"]
    kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()
    return recon + kl_weight * kl
