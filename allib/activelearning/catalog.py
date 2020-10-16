from aenum import Enum

class ALCatalog:
    class Paradigm(Enum):
        POOLBASED = "Poolbased"
        ESTIMATOR = "Estimator"
        ENSEMBLE  = "Ensemble"

    class QueryType(Enum):
        RANDOM_SAMPLING = "RandomSampling"
        LEAST_CONFIDENCE = "LeastConfidence"
        NEAR_DECISION_BOUNDARY = "NearDecisionBoundary"
        MARGIN_SAMPLING = "MarginSampling"
        MOST_CERTAIN = "MostCertain"
        MAX_ENTROPY = "MaxEntropy"
        INTERLEAVE = "InterleaveAL"
        LABELMAXIMIZER = "LabelMaximizer"
        MIN_ENTROPY = "MinEntropy"
        MOST_CONFIDENCE = "MostConfidence"