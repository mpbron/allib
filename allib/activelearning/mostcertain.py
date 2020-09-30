import numpy as np # type: ignore
from scipy.stats import entropy # type:ignore

from .ml_based import ProbabiltyBased, LabelProbabilityBased


class MostCertainSampling(ProbabiltyBased):
    _name = "MostCertainSampling"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        min_prob = np.amin(np.abs(prob_vec - 0.5), axis=1)
        return min_prob

class MinEntropy(ProbabiltyBased):
    _name = "MinEntropy"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        min_entropy = - entropy(prob_vec, axis=1)
        return min_entropy

class MostConfidence(ProbabiltyBased):
    _name = "MostConfidence"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        confidence = np.amax(prob_vec, axis=1)
        return confidence

class LabelMaximizer(LabelProbabilityBased):
    _name = "LabelMaximizer"
    @staticmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        return prob_vec

