from typing import Any, TypeVar

import numpy as np # type: ignore
from scipy.stats import entropy # type:ignore

from .catalog import ALCatalog
from .ml_based import AbstractSelectionCriterion
from .labelmethods import LabelProbabilityBased

class MostCertainSampling(AbstractSelectionCriterion):
    """Selects the training examples most far away from the 
    decision threshold at 0.50 / 50 % class probability.
    """    
    name = ALCatalog.QueryType.MOST_CERTAIN
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        min_prob: np.ndarray = np.amin(np.abs(prob_vec - 0.5), axis=1) # type: ignore
        return min_prob

class MinEntropy(AbstractSelectionCriterion):
    """Selects the training examples with the lowest entropy 
    at the probability level. This method is usable for
    Multilabel Classification.
    """    
    name = ALCatalog.QueryType.MIN_ENTROPY
    
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        min_entropy: np.ndarray = - entropy(prob_mat, axis=1)
        return min_entropy

class MostConfidence(AbstractSelectionCriterion):
    """Selects the training examples with the highest probability
    for **any** label in the probability matrix. 
    """
    name = ALCatalog.QueryType.MOST_CONFIDENCE
    
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        confidence: np.ndarray = np.amax(prob_mat, axis=1)
        return confidence

class LabelMaximizer(AbstractSelectionCriterion):
    """Identity function. This is usable for finding the 
    instance with the highest probability when the matrix is 
    sliced for one label only.

    TODO: Make the label a parameter 
    """
    name = ALCatalog.QueryType.LABELMAXIMIZER
    
    def __call__(self, prob_mat: np.ndarray) -> np.ndarray:
        return prob_mat

