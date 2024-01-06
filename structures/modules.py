import torch
from torch import nn
from typing import Optional

from .functions import *


class HyperbolicAlignmentModel(nn.Module):
    def __init__(self):
        super().__init__()
        # self.beta = nn.Parameter(torch.rand(1))
        # self.c = nn.Parameter(torch.rand(1))
        self.out = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_klein: bool = False,
        k_klein: bool = False
    ) -> torch.Tensor:
        q_h, k_h = hyperboloid(q, q_klein), hyperboloid(k, k_klein)
        d = dist_h(q_h, k_h)

        d = d.unsqueeze(-1)
        v = self.out(d).squeeze(-1)
        v = 1 / (v + EPS)
        if v.isnan().any():
            breakpoint()
        return v
        # return -self.beta * v - self.c


class HyperbolicAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.alignment = HyperbolicAlignmentModel()

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        v_k = klein(v)

        alpha = self.alignment(q, k)
        alpha = torch.exp(alpha)
        if mask_k is not None:
            alpha = alpha * mask_k.unsqueeze(-2)

        att = einstein_midpoint(alpha, v_k)

        if mask_q is not None:
            att = att * mask_q.unsqueeze(-1)

        return att


class HyperbolicAggregation(nn.Module):
    def __init__(self):
        super().__init__()
        self.attention = HyperbolicAttention()
        self.alpha = nn.Parameter(torch.rand(1))

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        att = self.attention(q, k, v, mask_q, mask_k)
        weight = torch.tensor([[self.alpha, 1-self.alpha]]).to(att.device)
        agg = torch.cat([klein(q), att], dim=-2)
        agg = einstein_midpoint(weight, agg)
        return rev_klein(agg)


class HyperbolicPathSearch(nn.Module):
    def __init__(
        self,
        n_layer: int
    ):
        super().__init__()
        self.paths = nn.ModuleList([HyperbolicAggregation() for _ in range(n_layer)])

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask_q: Optional[torch.Tensor] = None,
        mask_k: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        att = q
        for layer in self.paths:
            att = layer(att, k, v, mask_q, mask_k)
        return att