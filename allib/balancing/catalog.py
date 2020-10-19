from enum import Enum

class BalancerCatalog:
    class Type(Enum):
        IDENTITY = "Identity"
        RANDOM_OVER_SAMPLING = "RandomOverSampling"
        UNDERSAMPLING = "UnderSampling"
        DOUBLE = "DoubleBalancer"