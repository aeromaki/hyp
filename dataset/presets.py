from enum import Enum

from .dataset_info import DatasetInfo
from .tree_matrices import *


class DatasetPreset(Enum):
    WOS_S = DatasetInfo(GRAPH_WOS_S, "aeromaki/WOS5736", 3)
    WOS_M = DatasetInfo(GRAPH_WOS_M, "aeromaki/WOS11967", 3)
    WOS_L = DatasetInfo(GRAPH_WOS_L, "aeromaki/WOS46985", 3)
    PRIV_01 = DatasetInfo(GRAPH_PRIV_01, "aeromaki/htc-private-01", 4)