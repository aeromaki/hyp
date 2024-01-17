import torch
from torch import nn, Tensor
from typing import Optional, Union

from .functions import *


class HyperbolicAlignmentModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.beta = nn.Parameter(torch.rand(1))
        self.c = nn.Parameter(torch.rand(1))

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        q_klein: bool = False,
        k_klein: bool = False
    ) -> torch.Tensor:
        q_h, k_h = hyperboloid(q, q_klein), hyperboloid(k, k_klein)
        d = dist_h(q_h, k_h)
        d = d.unsqueeze(-1)
        alpha = -self.beta * d - self.c
        return alpha


class HyperbolicAttention(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.alignment = HyperbolicAlignmentModel()

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask_q: Optional[Tensor] = None,
        mask_k: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        v_k = klein(v)

        alpha = self.alignment(q, k)
        alpha = torch.exp(alpha)
        if mask_k is not None:
            alpha = alpha * mask_k.unsqueeze(-2)
        if mask_tgt is not None:
            alpha[mask_tgt] = -1000

        att = einstein_midpoint(alpha, v_k)
        if mask_q is not None:
            att = att * mask_q.unsqueeze(-1)

        return att


class AttentionProjector(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_proj: int,
        n_head: int
    ) -> None:
        self.proj = nn.Linear(d_model, d_proj*n_head)
        self.d_proj = d_proj
        self.n_head = n_head

    def forward(self, x: Tensor) -> Tensor:
        p = self.proj(x)
        p = p.view(*p.shape[:-2], self.n_head, p.shape[-2], self.d_proj)
        return p


class HyperbolicMHA(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int
    ) -> None:
        super().__init__()
        self.n_head = n_head
        self.Wq = AttentionProjector(d_model, d_k, n_head)
        self.Wk = AttentionProjector(d_model, d_k, n_head)
        self.Wv = AttentionProjector(d_model, d_v, n_head)
        self.attention = HyperbolicAttention()
        self.Wo = nn.Linear(d_v*n_head, d_model)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        mask_q: Optional[Tensor] = None,
        mask_k: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        q_mh = self.Wq(q)
        k_mh = self.Wk(k)
        v_mh = self.Wv(v)
        mask_q_mh = self._generate_mask(mask_q)
        mask_k_mh = self._generate_mask(mask_k)

        att = self.attention(q_mh, k_mh, v_mh, mask_q_mh, mask_k_mh, mask_tgt)
        res = self._reshape_v(att)
        return res

    def _generate_mask(self, mask: Union[NoneType, Tensor]) -> Tensor:
        if mask is not None:
            batch_ones = [1] * len(mask.shape[:-1])
            mask_mh = mask.unsqueeze(1).repeat(*batch_ones, self.n_head, 1)
        else:
            mask_mh = None
        return mask_mh

    def _reshape_v(self, output: Tensor) -> Tensor:
        res = output.transpose(-3, -2).flatten(-2)
        return res


class HyperbolicDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int
    ) -> None:
        self.mha = HyperbolicMHA(d_model, d_k, d_v, n_head)
        self.layernorm = nn.LayerNorm()

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        mask_q: Optional[Tensor] = None,
        mask_k: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        res = self.mha(q, k, k, mask_q, mask_k, mask_tgt)
        res = self.layernorm(src + res)
        return res


class HyperbolicDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int,
        n_layer: int
    ) -> None:
        self.layers = nn.ModuleList([
            HyperbolicDecoderLayer(d_model, d_k, d_v, n_head) for _ in range(n_layer)
        ])

    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        mask_q: Optional[Tensor] = None,
        mask_k: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        h = src
        for layer in self.layers:
            h = layer(h, tgt, mask_q, mask_k, mask_tgt)
        return h