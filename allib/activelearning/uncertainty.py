from typing import Generic, TypeVar, Any

import numpy as np # type: ignore
from scipy.stats import entropy #type: ignore

from .ml_based import ProbabiltyBased, LabelProbabilityBased

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")

class LeastConfidence(ProbabiltyBased[KT, DT, RT, LT], Generic[KT, DT, RT, LT]):
    _name = "LeastConfidence"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        max_prob = prob_vec.max(axis=1) # type: ignore
        return 1 - max_prob

class NearDecisionBoundary(ProbabiltyBased[KT, DT, RT, LT], Generic[KT, DT, RT, LT]):
    _name = "NearDecisionBoundary"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        min_prob = np.abs(prob_vec - 0.5).min(axis=1) # type: ignore
        return -1.0 * min_prob # type: ignore

class MarginSampling(ProbabiltyBased[KT, DT, RT, LT], Generic[KT, DT, RT, LT]):
    _name = "MarginSampling"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        # select the two highest probabilities per instance
        # with the lowest of the two on [:,0] and the highest on [:,1]
        two_best_prob = np.sort(prob_vec, axis=1)[:,-2:] # type: ignore
        # subtract
        margin = np.diff(two_best_prob, axis=1)
        # invert in order to use as a maximization method
        return -margin
    

class EntropySampling(ProbabiltyBased[KT, DT, RT, LT], Generic[KT, DT, RT, LT]):
    _name = "EntropySampling"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        return entropy(prob_vec, axis=1)

class LabelUncertainty(LabelProbabilityBased[KT, DT, RT, LT], Generic[KT, DT, RT, LT]):
    _name = "LabelUncertainty"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        min_prob = - np.abs(prob_vec - 0.5)
        return min_prob
