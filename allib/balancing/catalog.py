from enum import Enum

class BalancerType(Enum):
    IDENTITY = "Identity"
    RANDOM_OVER_SAMPLING = "RandomOverSampling"
    UNDERSAMPLING = "UnderSampling"
    DOUBLE = "DoubleBalancer"