import torch
from torch import nn
from typing import Optional
from transformers import AutoModel

from structures import MiniAlignDecoder


class MiniAlignModel(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        n_label: int,
        d_model: int,
        d_h: int,
        d_hh: int,
        n_head: int,
        n_layer: int,
        max_depth: int,
        label_graph: torch.Tensor,
        encoder_device: str = "cpu",
        decoder_device: str = "cpu",
        pretrained_label_embedding: Optional[nn.Parameter] = None
    ):
        super().__init__()

        self.device_e = encoder_device
        self.device_d = decoder_device
        self.n_label = n_label
        self.max_depth = max_depth
        self.label_graph = label_graph.to(self.device_d)

        self.encoder = self._init_encoder(encoder_name, encoder_device)
        self.decoder = MiniAlignDecoder(
            d_model=d_model,
            n_label=n_label,
            d_h=d_h,
            d_hh=d_hh,
            n_head=n_head,
            n_layer=n_layer,
            pretrained_label_embedding=pretrained_label_embedding
        ).to(self.device_d)

    def _init_encoder(
        cls,
        encoder_name: str,
        encoder_device: str
    ) -> nn.Module:
        encoder = AutoModel.from_pretrained(encoder_name)\
                                .eval()\
                                .to(encoder_device)
        return encoder

    def _get_encoder_last_hidden_state(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        input_ids = input_ids.to(self.device_e)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device_e)

        with torch.no_grad():
            e = self.encoder(input_ids, attn_mask).last_hidden_state\
                                                .cpu().to(self.device_d)

        return e

    def _switch_device_to_d(
        self,
        x: torch.Tensor
    ) -> torch.Tensor:
        return x.cpu().to(self.device_d)

    def _predict(
        self,
        e: torch.Tensor,
        attn_mask: Optional[torch.Tensor],
        decoder_input: Optional[torch.Tensor]
    ) -> torch.Tensor:
        # train
        if decoder_input is not None:
            b_e = e.repeat(self.max_depth-1, 1, 1)
            b_attn_mask = attn_mask.repeat(self.max_depth-1, 1)
            b_d = decoder_input.reshape(-1, 1)
            b_d_mask = self.label_graph[b_d].squeeze(1)

            out = self.decoder(
                b_e, b_d,
                mask_text=b_attn_mask,
                mask_label=b_d_mask
            )
            pred = out.reshape(-1, self.n_label)

        # inference
        else:
            e = self.decoder.embed_text(e)

            out = None
            pred = []
            for _ in range(self.max_depth):
                d = out.argmax(dim=-1) if out is not None\
                    else torch.zeros(e.shape[0], 1).to(self.device_d)
                out = self.decoder(
                    e, d,
                    mask_text=attn_mask,
                    mask_label=self.label_graph[d].squeeze(1)
                )
                pred.append(out)
            pred = torch.concat(pred, dim=-1)

        return pred

    def forward(
        self,
        input_ids: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
        decoder_input: Optional[torch.Tensor] = None
    ):
        e = self._get_encoder_last_hidden_state(input_ids, attn_mask)
        e = self._switch_device_to_d(e)
        if attn_mask is not None:
            attn_mask = self._switch_device_to_d(attn_mask)

        e = e[:, 0, :].unsqueeze(1)
        attn_mask = attn_mask[:, 0].unsqueeze(1)

        pred = self._predict(e, attn_mask, decoder_input)
        return pred