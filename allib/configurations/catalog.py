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
    CHAO_ENSEMBLE = "Chao_ENSEMBLE"
    AUTOTAR = "AUTOTAR"
    AUTOSTOP = "AUTOSTOP"
    CHAO_AT_ENSEMBLE = "CHAO_AT_ENSEMBLE"
    CHAO_IB_ENSEMBLE = "CHAO_IB_ENSEMBLE"
    TARGET = "TARGET"
    PRIOR = "PRIOR"


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
    LOGLINEAR = "Loglinear"


class StopBuilderConfiguration(str, Enum):
    CHAO_CONS_OPT = "Chao_CONS_OPT"
    CHAO_CONS_OPT_ALT = "Chao_CONS_OPT_ALT"
    CHAO_BOTH = "CHAOBOTH"
    RCAPTURE_ALL = "RCAPTURE_ALL"
    AUTOTAR = "AUTOTAR"
    AUTOSTOP = "AUTOSTOP"
    TARGET = "Target"


class ExperimentCombination(str, Enum):
    CHAO = "CHAO"
    CHAO_ALT = "CHAO_ALT"
    CHAO_BOTH = "CHAO_BOTH"
    AUTOTAR = "AUTOTAR"
    AUTOSTOP = "AUTOSTOP"
    CHAO_AT = "CHAO_AT"
    CHAO_IB = "CHAO_IB"
    RCAPTURE = "RCAPTURE"
    TARGET = "TARGET"
    PRIOR = "PRIOR"
