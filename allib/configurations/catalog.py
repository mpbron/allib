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
    AUTOSTOP = "AUTOSTOP"


class INITConfiguration(Enum):
    IDENTITY = "IDENTITY"
    RANDOM = "RANDOM"
    UNIFORM = "UNIFORM"
    SEPARATE = "SEPARATE"
    POSITVEUNIFORM = "POSITIVEUNIFORM"


class FEConfiguration(Enum):
    TFIDF5000 = "TfIDF5000"
    TFIDFTAR = "TFIDFTAR"


class EstimationConfiguration(str, Enum):
    RaschRidge = "RaschRidge"
    RaschParametric = "RaschParametric"
    RaschApproxParametric = "RaschApproxParametric"
    RaschApproxConvParametric = "RaschApproxConvParametric"
    CHAO = "Chao"
    AUTOSTOP = "AUTOSTOP"


class StopBuilderConfiguration(str, Enum):
    CHAO_CONS_OPT = "Chao_CONS_OPT"
    AUTOTAR = "AUTOTAR"
    AUTOSTOP = "AUTOSTOP"


class ExperimentCombination(str, Enum):
    CHAO4 = "CHAO4"
    AUTOTAR = "AUTOTAR"
    AUTOSTOP = "AUTOSTOP"
