from typing import Any, TypeVar

import numpy as np # type: ignore
from scipy.stats import entropy # type:ignore

from .catalog import ALCatalog
from .ml_based import ProbabiltyBased, LabelProbabilityBased

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")
class MostCertainSampling(ProbabiltyBased[KT, DT, VT, RT, LT, LVT, PVT]):
    _name = ALCatalog.QueryType.MOST_CERTAIN
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        min_prob: np.ndarray = np.amin(np.abs(prob_vec - 0.5), axis=1) # type: ignore
        return min_prob

class MinEntropy(ProbabiltyBased[KT, DT, VT, RT, LT, LVT, PVT]):
    _name = ALCatalog.QueryType.MIN_ENTROPY
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        min_entropy: np.ndarray = - entropy(prob_vec, axis=1)
        return min_entropy

class MostConfidence(ProbabiltyBased[KT, DT, VT, RT, LT, LVT, PVT]):
    _name = ALCatalog.QueryType.MOST_CONFIDENCE
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        confidence: np.ndarray = np.amax(prob_vec, axis=1)
        return confidence

class LabelMaximizer(LabelProbabilityBased[KT, DT, VT, RT, LT, LVT, PVT]):
    _name = ALCatalog.QueryType.LABELMAXIMIZER
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        return prob_vec

