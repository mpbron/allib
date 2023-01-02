from typing import Any, Dict
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
}

autostop = {
    "paradigm": Cat.AL.Paradigm.CUSTOM,
    "method": Cat.AL.CustomMethods.AUTOSTOP,
    "machinelearning": LR,
    "k_sample": 100,
    "batch_size": 1,
}
