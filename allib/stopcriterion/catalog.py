from enum import Enum

class StopCriterionCatalog(Enum):
    PREDICTION_ONLY = "PredictionOnly"
    INTERSECTION_FALLBACK = "IntersectionFallback"
    UPPERBOUND95 = "UpperBound95"