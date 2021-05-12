from allib.utils.random import get_random_generator
from typing import Generic, Optional, TypeVar, Any

import numpy as np # type: ignore
from scipy.stats import entropy #type: ignore

from .ml_based import AbstractSelectionCriterion
from .labelmethods import LabelProbabilityBased
from .catalog import ALCatalog

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")

class LeastConfidence(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.LEAST_CONFIDENCE
    
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        max_prob = prob_mat.max(axis=1) # type: ignore
        return 1 - max_prob # type: ignore

class NearDecisionBoundary(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.NEAR_DECISION_BOUNDARY
    
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        min_prob = np.abs(prob_vec - 0.5).min(axis=1) # type: ignore
        return -1.0 * min_prob # type: ignore

class MarginSampling(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.MARGIN_SAMPLING
    
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        # select the two highest probabilities per instance
        # with the lowest of the two on [:,0] and the highest on [:,1]
        two_best_prob = np.sort(prob_mat, axis=1)[:,-2:] # type: ignore
        # subtract
        margin = np.diff(two_best_prob, axis=1)
        # invert in order to use as a maximization method
        return -margin
class EntropySampling(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.MAX_ENTROPY
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        return entropy(prob_mat, axis=1)

class LabelUncertainty(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.LABELUNCERTAINTY

    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        min_prob = - np.abs(prob_mat - 0.5)
        return min_prob

class LabelUncertaintyNew(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.LABELUNCERTAINTY_NEW
    def __init__(self, label_column: int):
        super().__init__()
        self.label_column = label_column

    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        prob_mat_sliced = prob_mat[:,self.label_column]
        min_prob = - np.abs(prob_mat_sliced - 0.5)
        return min_prob

class RandomMLStrategy(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.RANDOM_ML

    def __init__(self, *_, rng: Optional[np.random.Generator] = None):
        self.rng = get_random_generator(rng)

    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        prob_mat_size = prob_mat.shape[0]
        random_score = self.rng.random((prob_mat_size,))
        return random_score



