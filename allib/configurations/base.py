from dataclasses import dataclass, field
import itertools
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy as np

from instancelib.utils.func import value_map, list_unzip, flatten_dicts
from ..estimation.base import AbstractEstimator
from ..estimation.rasch_comb_parametric import EMRaschRidgeParametricPython
from ..estimation.rasch_multiple import EMRaschRidgeParametricConvPython
from ..estimation.rasch_parametric import ParametricRaschPython
from ..estimation.rasch_python import EMRaschRidgePython
from ..estimation.mhmodel import AbundanceEstimator
from ..stopcriterion.base import AbstractStopCriterion
from ..stopcriterion.catalog import StopCriterionCatalog
from ..stopcriterion.estimation import (
    CombinedStopCriterion,
    Conservative,
    Optimistic,
    UpperboundCombinedCritertion,
)

from ..typehints import LT
from .catalog import (
    ALConfiguration,
    EstimationConfiguration,
    ExperimentCombination,
    FEConfiguration,
    StopBuilderConfiguration,
)
from .ensemble import (
    al_config_ensemble_prob,
    al_config_entropy,
    naive_bayes_estimator,
    rasch_estimator,
    rasch_lr,
    rasch_nblrrflgbm,
    rasch_rf,
    rasch_nblrrf,
    rasch_nblrrflgbmrand,
    svm_estimator,
    tf_idf5000,
    rasch_nblrrfsvm,
)

_K = TypeVar("_K")
_T = TypeVar("_T")
_U = TypeVar("_U")

AL_REPOSITORY = {
    ALConfiguration.NaiveBayesEstimator: naive_bayes_estimator,
    ALConfiguration.SVMEstimator: svm_estimator,
    ALConfiguration.RaschEstimator: rasch_estimator,
    ALConfiguration.EntropySamplingNB: al_config_entropy,
    ALConfiguration.ProbabilityEnsemble: al_config_ensemble_prob,
    ALConfiguration.RaschLR: rasch_lr,
    ALConfiguration.RaschNBLRRF: rasch_nblrrf,
    ALConfiguration.RaschNBLRRFSVM: rasch_nblrrfsvm,
    ALConfiguration.RaschRF: rasch_rf,
    ALConfiguration.RaschNBLRRFLGBM: rasch_nblrrflgbm,
    ALConfiguration.RaschNBLRRFLGBMRAND: rasch_nblrrflgbmrand,
}

FE_REPOSITORY = {FEConfiguration.TFIDF5000: tf_idf5000}

ESTIMATION_REPOSITORY = {
    EstimationConfiguration.RaschRidge: EMRaschRidgePython[
        int, str, np.ndarray, str, str
    ](),
    EstimationConfiguration.RaschParametric: ParametricRaschPython[
        int, str, np.ndarray, str, str
    ](),
    EstimationConfiguration.RaschApproxParametric: EMRaschRidgeParametricPython[
        int, str, np.ndarray, str, str
    ](),
    EstimationConfiguration.RaschApproxConvParametric: EMRaschRidgeParametricConvPython[
        int, str, np.ndarray, str, str
    ](),
    EstimationConfiguration.CHAO: AbundanceEstimator[Any, Any, Any, Any, Any, str](),
}

STOP_REPOSITORY: Dict[
    StopCriterionCatalog, Callable[[AbstractEstimator, Any], AbstractStopCriterion]
] = {
    StopCriterionCatalog.INTERSECTION_FALLBACK: lambda est, label: CombinedStopCriterion(
        est, label, 3, 1.0, 0.01
    ),
    StopCriterionCatalog.UPPERBOUND95: lambda est, label: UpperboundCombinedCritertion(
        est, label, 3, 1.0, 0.01
    ),
}


def filter_mapping(mapping: Mapping[_K, Optional[_T]]) -> Mapping[_K, _T]:
    return {k: v for k, v in mapping.items() if v is not None}


def key_map(f: Callable[[_K], _U], mapping: Mapping[_K, _T]) -> Mapping[_U, _T]:
    return {f(k): v for k, v in mapping.items()}


def mapping_unzip(
    mapping: Mapping[_K, Tuple[_T, _U]]
) -> Tuple[Mapping[_K, _T], Mapping[_K, _U]]:
    left_dict = {k: v for k, (v, _) in mapping.items()}
    right_dict = {k: v for k, (_, v) in mapping.items()}
    return left_dict, right_dict


@dataclass(frozen=True)
class TarExperimentParameters(Generic[LT]):
    al_configuration: ALConfiguration
    fe_configuration: FEConfiguration
    stop_builder_configuration: Sequence[StopBuilderConfiguration]
    batch_size: int
    stop_interval: int
    estimation_interval: int


def conservative_optimistic_builder(
    estimators: Mapping[str, AbstractEstimator], target: float
) -> Callable[
    [LT, LT],
    Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]],
]:
    def builder(
        pos_label: LT, neg_label: LT
    ) -> Tuple[
        Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]
    ]:
        conservatives = {
            f"{key}_conservative": Conservative.builder(est, target)(
                pos_label, neg_label
            )
            for key, est in estimators.items()
        }
        optimistics = {
            f"{key}_optimistic": Optimistic.builder(est, target)(pos_label, neg_label)
            for key, est in estimators.items()
        }
        return estimators, flatten_dicts(conservatives, optimistics)

    return builder


STOP_BUILDER_REPOSITORY = {
    StopBuilderConfiguration.CHAO_CONS_OPT: conservative_optimistic_builder(
        {"Chao": AbundanceEstimator()}, 0.95
    )
}


EXPERIMENT_REPOSITORY = {
    ExperimentCombination.CHAO4: TarExperimentParameters(
        ALConfiguration.RaschNBLRRFLGBMRAND,
        FEConfiguration.TFIDF5000,
        (StopBuilderConfiguration.CHAO_CONS_OPT,), 10, 10, 10
    )
}
