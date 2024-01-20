import torch
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer
from typing import Callable, Dict, List, Optional
from tqdm import tqdm
from sklearn.metrics import f1_score
from accelerate import Accelerator, DistributedDataParallelKwargs
import wandb

from dataset import Dataset
from model import Hypformer


class Trainer:
    def __init__(
        self,
        model: Hypformer,
        encoder_name: str,
        dataset: Dataset
    ) -> None:
        self.accelerator = Accelerator(
            split_batches=True,
            kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)]
        )

        self.encoder = AutoModel.from_pretrained(encoder_name)\
            .eval().to(self.accelerator.device)
        self.tokenizer = AutoTokenizer.from_pretrained(encoder_name)

        self.model = self.accelerator.prepare(model)

        self.dataset = dataset
        self.n_label = dataset.n_label
        self.graph = dataset.graph.to(self.accelerator.device)

        self.mask_label = None

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
        ).to(self.accelerator.device).values()
        attn_mask = attn_mask == 0

        decoder_input = labels[...,:-1]
        decoder_label = labels[...,1:]
        mask_tgt = Trainer._create_tgt_mask(decoder_input)
        mask_label = self.graph[decoder_input] == 0 if self.mask_label else None

        with torch.no_grad():
            e = self.encoder(input_ids, attn_mask).last_hidden_state
        logits = self.model(e, decoder_input, attn_mask, None, mask_tgt, mask_label)

        return logits, decoder_label

    def _init_weight(self, use_weight: bool) -> Tensor:
        if use_weight:
            count = torch.zeros(self.n_label)
            for i in self.dataset.dataset["train"]["label"]:
                count[i[1:]] += 1
            count[0] = 1
            weight = count.mean() / count
        else:
            weight = torch.ones(self.n_label)
        return weight.to(self.accelerator.device)

    def _create_dataset(self, which: str, batch_size: int) -> DataLoader:
        loader = self.dataset.create_loader(which, batch_size)
        loader = self.accelerator.prepare(loader)
        return loader

    def train(
        self,
        lr: float,
        batch_size: int,
        n_bb: int,
        n_print: int,
        n_val: int,
        n_save: int,
        n_iter: int,
        mask_label: bool,
        use_weight: bool,
        save_path: Callable[[int], str],
        config_wandb: Optional[Dict] = None
    ) -> None:
        self.mask_label = mask_label

        is_main = self.accelerator.is_main_process

        if config_wandb is not None and is_main:
            wandb.init(**config_wandb)
            log = lambda x: wandb.log(x)
        else:
            log = lambda x: None

        criterion = nn.CrossEntropyLoss(reduction="none")
        optimizer = optim.Adam(self.model.parameters(), lr=lr)

        optimizer = self.accelerator.prepare(optimizer)

        flat_cnt = 0
        loss_buffer = 0

        weight = self._init_weight(use_weight)

        for c in tqdm(range(n_iter)):
            # dataloader
            torch.manual_seed(c)
            train_loader = self._create_dataset("train", batch_size)

            # inner iteration
            for data in train_loader:
                # data
                texts, labels = data["text"], data["label"]
                labels = torch.stack(labels).T

                # step
                y_pred, labels = self._forward(texts, labels)
                y_pred = y_pred.reshape(-1, self.n_label)
                labels = labels.flatten()

                loss = criterion(y_pred, labels)
                loss = loss * weight[labels]
                loss = loss.mean()

                log({"loss": loss.item()})

                self.accelerator.backward(loss)

                # batch-batch
                flat_cnt += 1
                if flat_cnt % n_bb == 0:
                    optimizer.step()
                    optimizer.zero_grad()

                # print loss
                loss_buffer += loss.item()
                if flat_cnt % n_print == 0 and is_main:
                    loss_mean = loss_buffer / n_print
                    print(f"{flat_cnt: 6}: {loss_mean}")
                    log({"loss_mean": loss_mean})
                    loss_buffer = 0

                # validation
                if flat_cnt % n_val == 0 and is_main:
                    self.model.eval()
                    with torch.no_grad():
                        val_loader = self._create_dataset("validation", batch_size)
                        val_loader = self.accelerator.prepare(val_loader)
                        macro, micro = self._val(val_loader)
                        print(f"macro {macro}, micro {micro}")
                        log({"macro": macro, "micro": micro})
                    self.model.train()

                # save model (outer iteration)
                if flat_cnt % n_save == 0 and is_main:
                    self.accelerator.save_model(self.model, save_path(flat_cnt))
                    print("Model saved!")

        if flat_cnt % n_save != 0 and is_main:
            self.accelerator.save_model(self.model, save_path(flat_cnt))
            print("Model saved!")

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