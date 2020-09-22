from enum import Enum

class ALParadigm(Enum):
    POOLBASED = "Poolbased"

class QueryType(Enum):
    RANDOM_SAMPLING = "RandomSampling"
    LEAST_CONFIDENCE = "LeastConfidence"
    NEAR_DECISION_BOUNDARY = "NearDecisionBoundary"
    MARGIN_SAMPLING = "MarginSampling"
    MOST_CERTAIN = "MostCertain"
    MAX_ENTROPY = "MaxEntropy"
    VARIOUS = "VariousDjangoDefault"
    INTERLEAVE = "InterleaveAL"