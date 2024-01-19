import torch
from torch import nn, Tensor
from typing import Optional

from structures import HyperbolicDecoder, HyperbolicHead


class Hypformer(nn.Module):
    def __init__(
        self,
        d_encoder: int,
        d_eh: int,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int,
        d_ff: int,
        n_layer: int,
        n_label: int,
        max_depth: int
    ) -> None:
        super().__init__()
        self.labels = nn.Parameter(torch.rand(n_label, d_model))
        self.positional = nn.Parameter(torch.rand(max_depth, d_model))
        self.map = nn.Sequential(
            nn.Linear(d_encoder, d_eh),
            nn.GELU(),
            nn.Linear(d_eh, d_model)
        )
        self.decoder = HyperbolicDecoder(d_model, d_k, d_v, n_head, d_ff, n_layer)
        self.out = HyperbolicHead(d_model, d_k, n_head)

    def forward(
        self,
        bert_last_hidden_state: Tensor,
        decoder_input_ids: Tensor,
        mask_bert: Optional[Tensor] = None,
        mask_decoder: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None,
        mask_label: Optional[Tensor] = None
    ) -> Tensor:
        q = self.labels[decoder_input_ids] + self.positional[:decoder_input_ids.shape[-1]]
        k = self.map(bert_last_hidden_state)
        h = self.decoder(q, k, mask_decoder, mask_bert, mask_tgt)

        logits = self.out(h, self.labels)
        if mask_label is not None:
            logits[mask_label] = -1000

        return logits