from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, List, Mapping, Optional, Sequence, Tuple, Union
from uuid import UUID
from instancelib import TextInstance
from instancelib.instances.text import MemoryTextInstance
from instancelib.environment.text import TextEnvironment

from instancelib.ingest.qrel import TrecDataset, hidden
from numpy._typing import NDArray

from ..environment.base import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from .reviews import read_review_dataset
import numpy.typing as npt
from ..utils.io import create_dir_if_not_exists

class DatasetType(str, Enum):
    REVIEW = "Review"
    TREC = "Trec"

@dataclass
class TarDataset:
    type: DatasetType
    path: Path
    topic: Optional[str] = None

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

class TrecDatasetAL(TrecDataset):
    def get_env(self, topic_key: str) -> AbstractEnvironment[MemoryTextInstance[str, NDArray[Any]], str | UUID, str, NDArray[Any], str, str]:
        il_env = super().get_env(topic_key)
        al_env = MemoryEnvironment.from_instancelib_simulation(il_env)
        return al_env


@dataclass
class ReviewDatasets:
    paths: Mapping[str, Path]

    @property
    def topic_keys(self) -> Sequence[str]:
        return tuple(self.paths.keys()) 

    def get_env(
        self, topic_key: str
    ) -> AbstractEnvironment[TextInstance[int, npt.NDArray[Any]], Union[int,UUID], str, npt.NDArray[Any], str, str]:
        return read_review_dataset(self.paths[topic_key])

    def get_envs(
        self,
    ) -> Mapping[str, AbstractEnvironment[TextInstance[int, npt.NDArray[Any]], Union[int,UUID], str, npt.NDArray[Any], str, str]]:
        return {tk: self.get_env(tk) for tk in self.topic_keys}

    @classmethod
    def from_path(cls, base_dir: Path):
        paths = {
            f.stem: f
            for f in base_dir.iterdir()
            if not hidden(f) and "csv"  in f.suffix
        }
        return cls(paths)

def detect_type(path: Path) -> DatasetType:
    if (path / "qrels").exists():
        return DatasetType.TREC
    return DatasetType.REVIEW

