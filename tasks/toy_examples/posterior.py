import torch
import torch.nn as nn

from lampe.inference import NPE


class NPEEmbedding(nn.Module):
    def __init__(self, theta_dim, x_dim, embedding, **kwargs):
        super().__init__()
        self.npe = NPE(theta_dim=theta_dim, x_dim=x_dim, **kwargs)
        self.embedding = embedding

    def forward(self, theta: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.npe(theta, self.embedding(x))

    def flow(self, x: torch.Tensor):  # -> Distribution
        return self.npe.flow(self.embedding(x))
