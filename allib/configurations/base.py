from allib.stopcriterion.base import AbstractStopCriterion
from allib.estimation.base import AbstractEstimator
from typing import Any, Callable, Dict
import numpy as np

from ..stopcriterion.estimation import CombinedStopCriterion, UpperboundCombinedCritertion
from ..stopcriterion.catalog import StopCriterionCatalog
from ..estimation.rasch_parametric import ParametricRaschPython
from ..estimation.rasch_python import EMRaschRidgePython
from .catalog import ALConfiguration, FEConfiguration, EstimationConfiguration
from .ensemble import (al_config_ensemble_prob, al_config_entropy,
                       naive_bayes_estimator, rasch_estimator, rasch_lr,
                       rasch_rf, svm_estimator, tf_idf5000)

from ..typehints import LT

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
}

STOP_REPOSITORY: Dict[StopCriterionCatalog, Callable[[AbstractEstimator, Any], AbstractStopCriterion]] = {
   StopCriterionCatalog.INTERSECTION_FALLBACK: lambda est, label: CombinedStopCriterion(est, label, 3, 1.0, 0.01),
   StopCriterionCatalog.UPPERBOUND95: lambda est, label: UpperboundCombinedCritertion(est, label, 3, 1.0, 0.01) 
}
