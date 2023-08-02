from typing import Any, Dict
from ..estimation.catalog import EstimatorCatalog

from ..stopcriterion.catalog import StopCriterionCatalog
from ..module import ModuleCatalog as Cat
from .ensemble import tar_classifier, tf_idf_autotar

LR = tar_classifier(
    Cat.ML.SklearnModel.LOGISTIC,
    {
        "solver": "lbfgs",
        "C": 1.0,
        "max_iter": 10000,
    },
    tf_idf_autotar,
)

autotar = {
    "paradigm": Cat.AL.Paradigm.CUSTOM,
    "method": Cat.AL.CustomMethods.AUTOTAR,
    "machinelearning": LR,
    "k_sample": 100,
    "batch_size": 20,
    "initializer": lambda x: x,
}

autostop = {
    "paradigm": Cat.AL.Paradigm.CUSTOM,
    "method": Cat.AL.CustomMethods.AUTOSTOP,
    "machinelearning": LR,
    "k_sample": 100,
    "batch_size": 1,
}


def autostop_large(
    stop_criterion: StopCriterionCatalog,
    estimator: EstimatorCatalog,
    target_recall: float,
):
    return {
        "paradigm": Cat.AL.Paradigm.CUSTOM,
        "method": Cat.AL.CustomMethods.AUTOSTOP_LARGE,
        "machinelearning": LR,
        "k_sample": 100,
        "batch_size": 1,
        "estimator": estimator,
        "stopcriterion": stop_criterion,
        "target": target_recall,
    }
