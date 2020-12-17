from typing import Any, TypeVar

import numpy as np # type: ignore
from scipy.stats import entropy # type:ignore

from .catalog import ALCatalog
from .ml_based import AbstractSelectionCriterion
from .labelmethods import LabelProbabilityBased

class MostCertainSampling(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.MOST_CERTAIN
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        min_prob: np.ndarray = np.amin(np.abs(prob_vec - 0.5), axis=1) # type: ignore
        return min_prob

class MinEntropy(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.MIN_ENTROPY
    
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        min_entropy: np.ndarray = - entropy(prob_mat, axis=1)
        return min_entropy

class MostConfidence(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.MOST_CONFIDENCE
    
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        confidence: np.ndarray = np.amax(prob_mat, axis=1)
        return confidence

class LabelMaximizer(AbstractSelectionCriterion):
    name = ALCatalog.QueryType.LABELMAXIMIZER
    
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        return prob_mat

