from allib.activelearning.uncertainty import LabelUncertainty
from enum import Enum

class ALCatalog:
    class Paradigm(Enum):
        POOLBASED = "Poolbased"
        ESTIMATOR = "Estimator"
        CYCLE = "CycleEstimator"
        ENSEMBLE  = "Ensemble"
        NEWESTIMATOR = "NewEstimator"
        PROBABILITY_BASED = "ProbabilityBased"

    class QueryType(str, Enum):
        RANDOM_SAMPLING = "RandomSampling"
        LEAST_CONFIDENCE = "LeastConfidence"
        NEAR_DECISION_BOUNDARY = "NearDecisionBoundary"
        MARGIN_SAMPLING = "MarginSampling"
        MOST_CERTAIN = "MostCertain"
        MAX_ENTROPY = "MaxEntropy"
        INTERLEAVE = "InterleaveAL"
        LABELMAXIMIZER = "LabelMaximizer"
        LABELUNCERTAINTY = "LabelUncertainty"
        MIN_ENTROPY = "MinEntropy"
        MOST_CONFIDENCE = "MostConfidence"
        PRELABELED = "Prelabeled"