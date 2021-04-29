from enum import Enum

class ALConfiguration(Enum):
    NaiveBayesEstimator = "NaiveBayesEstimator"
    SVMEstimator = "SVMEstimator"
    MixedEstimator = "MixedEstimator"
    RaschEstimator = "RaschEstimator"

class FEConfiguration(Enum):
    TFIDF5000 = "TfIDF5000"