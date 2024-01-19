import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from typing import Optional, List, Any, Callable, Dict
from tqdm import tqdm
from sklearn.metrics import f1_score
import wandb

from dataset import Dataset
from model import Hypformer


class EncoderContainer:
    def __init__(
        self,
        encoder_name: str,
        device_e: str = "cpu",
        device_d: str = "cpu"
    ) -> None:
        self.encoder = EncoderContainer._init_encoder(encoder_name, device_e)
        self.device_e = device_e
        self.device_d = device_d
        self._switch_device_to_d = (lambda x: x.cpu().to(device_d))\
            if device_e != device_d\
            else (lambda x: x)

    def __call__(
        self,
        model: Hypformer,
        input_ids: Tensor,
        decoder_input: Tensor,
        attn_mask: Optional[Tensor] = None,
        decoder_mask: Optional[Tensor] = None,
        mask_tgt: Optional[Tensor] = None
    ) -> Tensor:
        e = self._get_encoder_last_hidden_state(input_ids, attn_mask)
        e = self._switch_device_to_d(e)
        if attn_mask is not None:
            attn_mask = self._switch_device_to_d(attn_mask)
        pred = model(e, decoder_input, attn_mask, decoder_mask, mask_tgt)
        return pred

    @staticmethod
    def _init_encoder(
        encoder_name: str,
        device_e: str
    ) -> Any:
        encoder = AutoModel.from_pretrained(encoder_name)\
                                .eval()\
                                .to(device_e)
        return encoder

    @torch.no_grad()
    def _get_encoder_last_hidden_state(
        self,
        input_ids: Tensor,
        attn_mask: Optional[Tensor]
    ) -> Tensor:
        input_ids = input_ids.to(self.device_e)
        if attn_mask is not None:
            attn_mask = attn_mask.to(self.device_e)
        last_hidden_state = self.encoder(input_ids, attn_mask).last_hidden_state
        return last_hidden_state


class Trainer:
    def __init__(
        self,
        model: Hypformer,
        encoder_name: str = "bert-base-uncased",
        device_e: str = "cpu",
        device_d: str = "cpu"
    ) -> None:
        self.container = EncoderContainer(encoder_name, device_e, device_d)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)
        self.model = model
        self.device_e = device_e
        self.device_d = device_d
        self.n_label = None

    @staticmethod
    def _create_tgt_mask(tgt: Tensor) -> Tensor:
        len_seq = tgt.shape[-1]
        ones = torch.ones(len_seq, len_seq, device=tgt.device)
        mask = torch.triu(ones, diagonal=1).type(torch.bool)
        return mask

    def _forward(
        self,
        texts: List[str],
        labels: Tensor
    ) -> (Tensor, Tensor):
        input_ids, _, attn_mask = self.tokenizer(
            texts,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512
        ).values()
        attn_mask = attn_mask == 0

        labels = labels.to(self.device_d)
        decoder_input = labels[...,:-1]
        decoder_label = labels[...,1:]
        mask_tgt = Trainer._create_tgt_mask(decoder_input)

        logits = self.container(
            model=self.model,
            input_ids=input_ids,
            decoder_input=decoder_input,
            attn_mask=attn_mask,
            mask_tgt=mask_tgt
        )
        return logits, decoder_label

    def _map_dataloader(self, data) -> torch.Tensor:
        texts, labels = data["text"], data["label"]
        labels = torch.stack(labels).T
        y_pred, labels = self._forward(texts, labels)
        return (y_pred.reshape(-1, self.n_label).argmax(dim=1), labels.flatten())

    @torch.no_grad()
    def _val(self, dataset: DataLoader) -> (float, float):
        preds = torch.tensor([])
        labels = torch.tensor([])

        for pred, label in map(self._map_dataloader, dataset):
            preds = torch.cat([preds, pred.cpu()])
            labels = torch.cat([labels, label.cpu()])

        macro = f1_score(preds, labels, average="macro")
        micro = f1_score(preds, labels, average="micro")
        return macro, micro

    def _init_weight(self, dataset: Any) -> Tensor:
        count = torch.zeros(self.n_label).to(self.device_d)
        for i in dataset.dataset["train"]["label"]:
            count[i[1:]] += 1
        count[0] = 1
        weight = count.mean() / count
        return weight

    def train(
        self,
        config_wandb: Dict,
        dataset: Dataset,
        lr: float,
        batch_size: int,
        n_bb: int,
        n_print: int,
        n_val: int,
        n_save: int,
        n_iter: int,
        save_path: Callable[[int], str]
    ) -> None:
        wandb.init(**config_wandb)

        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        flat_cnt = 0
        loss_buffer = 0

        self.n_label = dataset.n_label

        weight = self._init_weight(dataset)

        for c in tqdm(range(n_iter)):
            # dataloader
            torch.manual_seed(c)
            train_loader = dataset.create_loader("train", batch_size)

            # inner iteration
            for data in tqdm(train_loader):
                # data
                texts, labels = data["text"], data["label"]
                labels = torch.stack(labels).T

                # step
                y_pred, labels = self._forward(texts, labels)
                y_pred = y_pred.reshape(-1, dataset.n_label)
                labels = labels.flatten()

                loss = criterion(y_pred, labels)
                loss = loss * weight[labels]
                loss = loss.mean()

                wandb.log({"loss": loss.item()})

                try:
                    loss.backward()
                except:
                    torch.save({"model": self.model.state_dict()}, save_path(flat_cnt))
                    breakpoint()

                # batch-batch
                flat_cnt += 1
                if flat_cnt % n_bb == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # print loss
                loss_buffer += loss.item()
                if flat_cnt % n_print == 0:
                    loss_mean = loss_buffer / n_print
                    wandb.log({"loss_mean": loss_mean})
                    print("\n", loss_mean)
                    loss_buffer = 0

                # validation
                if flat_cnt % n_val == 0:
                    self.model.eval()
                    with torch.no_grad():
                        macro, micro = self._val(dataset.create_loader("validation", batch_size))
                        print(f"macro {macro}, micro {micro}")
                        wandb.log({"macro": macro, "micro": micro})
                    self.model.train()

                # save model (outer iteration)
                if flat_cnt % n_save == 0:
                    torch.save({"model": self.model.state_dict()}, save_path(flat_cnt))
                    print("Model saved!")
