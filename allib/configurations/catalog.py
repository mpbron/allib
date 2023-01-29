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
    ILRaschNBLRRFLGBMRAND = "ILRaschNBLRRFLGBMRAND"
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
    CHAO_CONS_OPT_ALT = "Chao_CONS_OPT_ALT"
    CHAO_BOTH = "CHAOBOTH"
    AUTOTAR = "AUTOTAR"
    AUTOSTOP = "AUTOSTOP"


class ExperimentCombination(str, Enum):
    CHAO = "CHAO"
    CHAO_ALT = "CHAO_ALT"
    CHAO_BOTH = "CHAO_BOTH"
    AUTOTAR = "AUTOTAR"
    AUTOSTOP = "AUTOSTOP"
