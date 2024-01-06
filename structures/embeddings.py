import torch
from torch import nn

from .functions import EPS


class HypTextEmbedding_Normed(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_h: int
    ):
        super().__init__()
        self.hyp_map = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_h)
        )

    def forward(
        self,
        e: torch.Tensor
    ) -> torch.Tensor:
        e = self.hyp_map(e)
        e = e / (e.norm(dim=-1, keepdim=True) + EPS)
        return e


class HypTextEmbedding(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_h: int
    ):
        super().__init__()
        self.hyp_map = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_h)
        )

    def forward(
        self,
        e: torch.Tensor
    ) -> torch.Tensor:
        e = self.hyp_map(e)
        return e