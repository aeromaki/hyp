import torch
from torch import nn

from .functions import *
from .modules import *
from .embeddings import HypTextEmbedding, HypTextEmbedding_Normed

class MiniAlignDecoder(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_label: int,
        d_h: int,
        d_hh: int,
        n_head: int,
        n_layer: int,
        pretrained_label_embedding: Optional[nn.Parameter] = None
    ):
        super().__init__()

        if pretrained_label_embedding is not None:
            self.label = pretrained_label_embedding #.detach()
        else:
            self.label = nn.Parameter(torch.rand(n_label, d_h))

        self.n_head = n_head
        self.d_hh = d_hh
        self.textconv = nn.Sequential(
            nn.Linear(d_model, d_model*2),
            nn.ReLU(),
            nn.Linear(d_model*2, d_hh*n_head)
        )
        self.text2hyp = HypTextEmbedding(d_hh, d_h)

        self.attention = HyperbolicPathSearch(n_layer)
        self.agg_p = nn.Sequential(
            nn.Linear(n_head, n_head*2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(n_head*2, n_head*2),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(n_head*2, n_head),
            nn.ReLU(),
            # nn.Dropout(0.1),
            nn.Linear(n_head, 1)
        )

    def embed_text(
        self,
        bert_last_hidden_state: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor):
        e = self.textconv(bert_last_hidden_state)
        e = e.view(e.shape[0], self.n_head, e.shape[1], self.d_hh)

        return self.text2hyp(e)

    def _amplify(
        self,
        e: torch.Tensor,
        mask_label: torch.Tensor
    ) -> torch.Tensor:
        valid = self.label.T.unsqueeze(0) * mask_label.unsqueeze(1)
        norm = valid.transpose(1, 2).norm(dim=2)
        norm_sum = norm.sum(-1)
        norm_mean = (norm_sum / mask_label.sum(-1))

        amp = norm_mean.sqrt()
        amp = amp.reshape(-1, 1, 1, 1)
        return e * amp

    def _dist(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        q_klein: bool = False,
        k_klein: bool = False
    ) -> torch.Tensor:
        q_h, k_h = hyperboloid(q, q_klein), hyperboloid(k, k_klein)
        d = dist_h(q_h, k_h)
        return d

    def forward(
        self,
        e: torch.Tensor, # encoder last hidden state | embedded
        y: torch.Tensor, # t-1 label
        mask_text: torch.Tensor,
        mask_label: torch.Tensor
    ) -> (torch.Tensor, torch.Tensor): # h_new, output
        """
        if len(e.shape) < 4:
            e = self.embed_text(e)
            e = self._amplify(e, mask_label)
        """
        ek = self.embed_text(e)
        # ev = self._amplify(ev, mask_label)

        q = self.label[y]
        q = q.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        mask_text = mask_text.unsqueeze(1).repeat(1, self.n_head, 1)

        att = self.attention(
            q=q,
            k=ek,
            v=ek,
            mask_k=mask_text
        )

        k = self.label.view(1, 1, *self.label.shape)
        dist = self._dist(att, k)

        dist = dist.squeeze().transpose(-2, -1)

        out = self.agg_p(dist)
        out = out.squeeze()
        out[mask_label==0] = -500

        if out.isnan().any():
            breakpoint()
        return out