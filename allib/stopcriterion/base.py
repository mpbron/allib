import collections
import itertools
from abc import ABC, abstractmethod
from typing import Any, Deque, Generic, TypeVar

import numpy as np  # type: ignore

from ..activelearning import ActiveLearner
from ..activelearning.ensembles import AbstractEnsemble
from ..activelearning.estimator import Estimator
from ..estimation.base import AbstractEstimator
from ..estimation.rcapture import AbundanceEstimator
from ..utils.func import all_equal, intersection
from ..analysis.analysis import process_performance

KT = TypeVar("KT")
VT = TypeVar("VT")
DT = TypeVar("DT")
RT = TypeVar("RT")
LT = TypeVar("LT")

class AbstractStopCriterion(ABC, Generic[LT]):
    @abstractmethod
    def update(self, learner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        pass

    @property
    @abstractmethod
    def stop_criterion(self) -> bool:
        pass

