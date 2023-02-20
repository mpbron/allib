from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional

from instancelib.ingest.qrel import TrecDataset

from ..environment.base import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from .reviews import read_review_dataset


class DatasetType(Enum):
    REVIEW = "Review"
    TREC = "Trec"


@dataclass
class TarDataset:
    type: DatasetType
    path: Path
    topic: Optional[str]

    @property
    def env(self) -> AbstractEnvironment:
        if self.type == DatasetType.REVIEW:
            return read_review_dataset(self.path)
        if self.type == DatasetType.TREC and self.topic is not None:
            trec = TrecDataset.from_path(self.path)
            il_env = trec.get_env(self.topic)
            al_env = MemoryEnvironment.from_instancelib_simulation(il_env)
            return al_env
        raise NotImplementedError("This combination is not yet implemented")
