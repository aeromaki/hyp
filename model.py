import torch
from torch import nn, Tensor

from structures import HyperbolicDecoder


class Hypformer(nn.Module):
    def __init__(
        self,
        d_encoder: int,
        d_model: int,
        d_k: int,
        d_v: int,
        n_head: int,
        n_layer: int,
        n_label: int,
        max_depth: int
    ) -> None:
        self.labels = nn.Embedding(n_label, d_model)
        self.positional = nn.Parameter(max_depth, d_model)
        self.proj = nn.Linear(d_encoder, d_model)
        self.decoder = HyperbolicDecoder(d_model, d_k, d_v, n_head, n_layer)
        self.out = nn.Linear(d_model, n_label)

    def forward(
        self,
        bert_last_hidden_state: Tensor,
        decoder_input_ids: Tensor,
        mask_bert: Optional[Tensor] = None,
        mask_decoder: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        q = self.labels(decoder_input_ids) + self.positional[:decoder_input_ids.shape[-1]]
        k = self.proj(bert_last_hidden_state)
        h = self.decoder(q, k, mask_decoder, mask_bert, mask_tgt)
        logits = self.out(h)
        return logits
