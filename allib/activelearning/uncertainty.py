import numpy as np # type: ignore
from scipy.stats import entropy #type: ignore

from .ml_based import ProbabiltyBased, LabelProbabilityBased

class LeastConfidence(ProbabiltyBased):
    _name = "LeastConfidence"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        max_prob = prob_vec.max(axis=1)
        return 1 - max_prob

class NearDecisionBoundary(ProbabiltyBased):
    _name = "NearDecisionBoundary"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        min_prob = - np.abs(prob_vec - 0.5).min(axis=1)
        return min_prob

class MarginSampling(ProbabiltyBased):
    _name = "MarginSampling"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        # select the two highest probabilities per instance
        # with the lowest of the two on [:,0] and the highest on [:,1]
        two_best_prob = np.sort(prob_vec, axis=1)[:,-2:]
        # subtract
        margin = np.diff(two_best_prob, axis=1)
        # invert in order to use as a maximization method
        return -margin
    

class EntropySampling(ProbabiltyBased):
    _name = "EntropySampling"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        return entropy(prob_vec, axis=1)

class LabelUncertainty(LabelProbabilityBased):
    name = "LabelUncertainty"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        min_prob = - np.abs(prob_vec - 0.5)
        return min_prob
