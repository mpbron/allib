import itertools
from abc import ABC, abstractmethod
from collections import Counter
from dataclasses import dataclass
from typing import (
    Any,
    FrozenSet,
    Generic,
    List,
    Optional,
    Sequence,
    Set,
    Tuple,
    TypeVar,
)

import numpy as np
import numpy.typing as npt
import pandas as pd  # type: ignore
from sklearn.linear_model import LogisticRegression  # type: ignore

from ..activelearning.base import ActiveLearner
from ..activelearning.ml_based import FeatureMatrix, MLBased
from ..environment import AbstractEnvironment
from ..machinelearning import AbstractClassifier
from ..typehints import DT, IT, KT, LT, RT, VT
from typing_extensions import Self


@dataclass
class Estimate:
    point: float
    lower_bound: float
    upper_bound: float

    @classmethod
    def empty(cls) -> Self:
        return cls(float("nan"), float("nan"), float("nan"))


class AbstractEstimator(ABC, Generic[IT, KT, DT, VT, RT, LT]):
    name = "AbstractEstimator"

    @abstractmethod
    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT], label: LT
    ) -> Estimate:
        raise NotImplementedError


@dataclass
class DecisionRow(Generic[KT]):
    key: KT
    probability: float
    labeled: bool
    pos_labeled: bool
    order: Optional[int]
