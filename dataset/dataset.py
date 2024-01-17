from typing import Any, Union
from datasets import load_dataset
from torch import tensor, Tensor
from torch.utils.data import DataLoader

from presets import DatasetPreset


class Dataset:
    def __init__(
        self,
        preset: Union[DatasetPreset, str]
    ) -> None:
        if isinstance(preset, str):
            preset = DatasetPreset[preset]
        graph, load_path, max_depth = preset.value.get_infos()

        self.graph = tensor(graph)
        self.n_label = len(graph.shape[0])
        self.max_depth = max_depth
        self.dataset = Dataset._load_dataset(load_path)

    def create_loader(
        self,
        which: str,
        batch_size: int
    ) -> DataLoader:
        loader = DataLoader(self.dataset[which], batch_size=batch_size, shuffle=True)
        return loader

    def _load_dataset(self, load_path: str) -> Any:
        dataset = load_dataset(load_path)
        dataset = dataset.map(self._preprocess_dataset)
        return dataset

    def _preprocess_dataset(self, row: Any) -> Any:
        row["label"] = tensor(row["label"] + [self.n_label-1] * (self.max_depth - len(row["label"])))
        return row