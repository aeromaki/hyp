from dataclasses import dataclass
from typing import List


@dataclass
class DatasetInfo:
    graph: List[List[int]]
    load_path: str
    max_depth: int

    def get_infos(self) -> (List[List[int]], str, int):
        return self.graph, self.load_path, self.max_depth