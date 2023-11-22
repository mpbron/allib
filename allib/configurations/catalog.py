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
    CHAO_IB_ENSEMBLE_20 = "CHAO_IB_ENSEMBLE_20"
    CHAO_SAME = "CHAO_SAME"
    TARGET = "TARGET"
    PRIOR = "PRIOR"
    AUTOSTOP_LARGE_CONS_95 = "AUTOSTOP_LARGE_CONS_95"
    AUTOSTOP_LARGE_CONS_100 = "AUTOSTOP_LARGE_CONS_100"
    AUTOSTOP_LARGE_OPT_95 = "AUTOSTOP_LARGE_OPT_95"
    AUTOSTOP_LARGE_OPT_100 = "AUTOSTOP_LARGE_OPT_100"
    CMH = "CMH"


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
    CHAO_LEAN = "Chao_LEAN"
    CHAO_CONS_OPT_ALT = "Chao_CONS_OPT_ALT"
    CHAO_BOTH = "CHAOBOTH"
    RCAPTURE_ALL = "RCAPTURE_ALL"
    AUTOTAR = "AUTOTAR"
    AUTOSTOP = "AUTOSTOP"
    TARGET = "Target"
    LASTSEQUENCE = "LastSequence"
    QUANT = "QUANT"
    CMH = "CMH"


class ExperimentCombination(str, Enum):
    CMH = "CMH"
    CHAO = "CHAO"
    CHAO_ALT = "CHAO_ALT"
    CHAO_BOTH = "CHAO_BOTH"
    AUTOTAR = "AUTOTAR"
    AUTOTAR_QUANT = "AUTOTAR_QUANT"
    AUTOSTOP = "AUTOSTOP"
    CHAO_AT = "CHAO_AT"
    CHAO_IB = "CHAO_IB"
    CHAO_IB_20 = "CHAO_IB_20"
    CHAO_SAME = "CHAO_SAME"
    RCAPTURE = "RCAPTURE"
    TARGET = "TARGET"
    PRIOR = "PRIOR"
    AUTOSTOP_LARGE_CONS_95 = "AUTOSTOP_LARGE"
    AUTOSTOP_LARGE_CONS_100 = "AUTOSTOP_LARGE_CONS_100"
    AUTOSTOP_LARGE_OPT_95 = "AUTOSTOP_LARGE_OPT_95"
    AUTOSTOP_LARGE_OPT_100 = "AUTOSTOP_LARGE_OPT_100"
