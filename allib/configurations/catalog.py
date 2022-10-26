from enum import Enum


class ALConfiguration(Enum):
    NaiveBayesEstimator = "NaiveBayesEstimator"
    SVMEstimator = "SVMEstimator"
    MixedEstimator = "MixedEstimator"
    RaschEstimator = "RaschEstimator"
    RaschLR = "RaschLR"
    RaschRF = "RaschRF"
    EntropySamplingNB = "EntropySamplingNB"
    ProbabilityEnsemble = "ProbabilityEnsemble"
    RaschNBLRRF = "RaschNBLRRF"
    RaschNBLRRFSVM = "RaschNBLRRFSVM"
    RaschNBLRRFLGBM = "RaschNBLRRFLBGM"
    RaschNBLRRFLGBMRAND = "RaschNBLRRFLGBMRAND"
    AUTOTAR = "AUTOTAR"


class FEConfiguration(Enum):
    TFIDF5000 = "TfIDF5000"


class EstimationConfiguration(str, Enum):
    RaschRidge = "RaschRidge"
    RaschParametric = "RaschParametric"
    RaschApproxParametric = "RaschApproxParametric"
    RaschApproxConvParametric = "RaschApproxConvParametric"
    CHAO = "Chao"


class StopBuilderConfiguration(str, Enum):
    CHAO_CONS_OPT = "Chao_CONS_OPT"
    AUTOTAR = "AUTOTAR"


class ExperimentCombination(str, Enum):
    CHAO4 = "CHAO4"
    AUTOTAR = "AUTOTAR"
