from __future__ import annotations
import torch
import torch.nn as nn


class EmbeddingNet(nn.Module):
    """
    Simple MLP that maps aggregated acoustic features to a 128-dim embedding.
    Input expects a 1D vector: concat([mfcc_mean, mfcc_std, rms_mean/std, centroid_mean/std]).
    """
    def __init__(self, in_dim: int, emb_dim: int = 128):
        super().__init__()
        hidden = 256
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.BatchNorm1d(hidden),
            nn.Dropout(0.1),
            nn.Linear(hidden, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = torch.nn.functional.normalize(z, p=2, dim=-1)
        return z


def save_model(model: EmbeddingNet, path: str):
    torch.save(model.state_dict(), path)


def load_model(path: str, in_dim: int, emb_dim: int = 128, map_location: str | None = None) -> EmbeddingNet:
    m = EmbeddingNet(in_dim=in_dim, emb_dim=emb_dim)
    sd = torch.load(path, map_location=map_location or 'cpu')
    m.load_state_dict(sd)
    m.eval()
    return m
