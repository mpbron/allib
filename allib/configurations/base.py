from typing import Any, Callable, Dict

import numpy as np

from ..estimation.base import AbstractEstimator
from ..estimation.rasch_comb_parametric import EMRaschRidgeParametricPython
from ..estimation.rasch_multiple import EMRaschRidgeParametricConvPython
from ..estimation.rasch_parametric import ParametricRaschPython
from ..estimation.rasch_python import EMRaschRidgePython
from ..stopcriterion.base import AbstractStopCriterion
from ..stopcriterion.catalog import StopCriterionCatalog
from ..stopcriterion.estimation import (CombinedStopCriterion,
                                        UpperboundCombinedCritertion)
from ..typehints import LT
from .catalog import ALConfiguration, EstimationConfiguration, FEConfiguration
from .ensemble import (al_config_ensemble_prob, al_config_entropy,
                       naive_bayes_estimator, rasch_estimator, rasch_lr,
                       rasch_rf, svm_estimator, tf_idf5000)

AL_REPOSITORY = {
    ALConfiguration.NaiveBayesEstimator :  naive_bayes_estimator,
    ALConfiguration.SVMEstimator : svm_estimator,
    ALConfiguration.RaschEstimator: rasch_estimator,
    ALConfiguration.EntropySamplingNB: al_config_entropy,
    ALConfiguration.ProbabilityEnsemble: al_config_ensemble_prob,
    ALConfiguration.RaschLR: rasch_lr,
    ALConfiguration.RaschRF: rasch_rf,
}

FE_REPOSITORY = {
    FEConfiguration.TFIDF5000 : tf_idf5000
}

ESTIMATION_REPOSITORY = {
    EstimationConfiguration.RaschRidge: EMRaschRidgePython[int, str, np.ndarray, str, str](),
    EstimationConfiguration.RaschParametric: ParametricRaschPython[int, str, np.ndarray, str, str](),
    EstimationConfiguration.RaschApproxParametric: EMRaschRidgeParametricPython[int, str, np.ndarray, str, str](),
    EstimationConfiguration.RaschApproxConvParametric: EMRaschRidgeParametricConvPython[int,str, np.ndarray, str, str](),
}

STOP_REPOSITORY: Dict[StopCriterionCatalog, Callable[[AbstractEstimator, Any], AbstractStopCriterion]] = {
   StopCriterionCatalog.INTERSECTION_FALLBACK: lambda est, label: CombinedStopCriterion(est, label, 3, 1.0, 0.01),
   StopCriterionCatalog.UPPERBOUND95: lambda est, label: UpperboundCombinedCritertion(est, label, 3, 1.0, 0.01) 
}
