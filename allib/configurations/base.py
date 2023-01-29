from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy.typing as npt
from instancelib.utils.func import flatten_dicts

from allib.analysis.initialization import (
    Initializer,
    RandomInitializer,
    SeparateInitializer,
)
from ..estimation.autostop import HorvitzThompsonVar2

from ..estimation.base import AbstractEstimator
from ..estimation.mhmodel import ChaoEstimator, ChaoAlternative
from ..estimation.rasch_comb_parametric import EMRaschRidgeParametricPython
from ..estimation.rasch_multiple import EMRaschRidgeParametricConvPython
from ..estimation.rasch_parametric import ParametricRaschPython
from ..estimation.rasch_python import EMRaschRidgePython
from ..stopcriterion.base import AbstractStopCriterion
from ..stopcriterion.estimation import (
    Conservative,
    Optimistic,
)
from ..stopcriterion.heuristic import AprioriRecallTarget
from ..stopcriterion.others import (
    BudgetStoppingRule,
    KneeStoppingRule,
    ReviewHalfStoppingRule,
    Rule2399StoppingRule,
    StopAfterKNegative,
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
    ilrasch_nblrrflgbmrand,
    naive_bayes_estimator,
    rasch_estimator,
    rasch_lr,
    rasch_nblrrf,
    rasch_nblrrflgbm,
    rasch_nblrrflgbmrand,
    rasch_nblrrfsvm,
    rasch_rf,
    svm_estimator,
    tf_idf5000,
)

from .tarbaselines import autotar, autostop

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
    ALConfiguration.ILRaschNBLRRFLGBMRAND: ilrasch_nblrrflgbmrand,
    ALConfiguration.AUTOTAR: autotar,
    ALConfiguration.AUTOSTOP: autostop,
}

FE_REPOSITORY = {FEConfiguration.TFIDF5000: tf_idf5000}

ESTIMATION_REPOSITORY = {
    EstimationConfiguration.RaschRidge: EMRaschRidgePython[
        int, str, npt.NDArray[Any], str, str
    ](),
    EstimationConfiguration.RaschParametric: ParametricRaschPython[
        int, str, npt.NDArray[Any], str, str
    ](),
    EstimationConfiguration.RaschApproxParametric: EMRaschRidgeParametricPython[
        int, str, npt.NDArray[Any], str, str
    ](),
    EstimationConfiguration.RaschApproxConvParametric: EMRaschRidgeParametricConvPython[
        int, str, npt.NDArray[Any], str, str
    ](),
    EstimationConfiguration.CHAO: ChaoEstimator[Any, Any, Any, Any, Any, str](),
    EstimationConfiguration.AUTOSTOP: HorvitzThompsonVar2(),
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
    fe_configuration: Optional[FEConfiguration]
    init_configuration: Callable[..., Initializer[Any, Any, LT]]
    stop_builder_configuration: Sequence[StopBuilderConfiguration]
    batch_size: int
    stop_interval: int
    estimation_interval: int


def conservative_optimistic_builder(
    estimators: Mapping[str, AbstractEstimator], targets: Sequence[float]
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
            f"{key}_conservative_{target}": Conservative.builder(est, target)(
                pos_label, neg_label
            )
            for key, est in estimators.items()
            for target in targets
        }
        optimistics = {
            f"{key}_optimistic_{target}": Optimistic.builder(est, target)(
                pos_label, neg_label
            )
            for key, est in estimators.items()
            for target in targets
        }
        return estimators, flatten_dicts(conservatives, optimistics)

    return builder


def combine_builders(
    a: Callable[
        [LT, LT],
        Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]],
    ],
    b: Callable[
        [LT, LT],
        Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]],
    ],
) -> Callable[
    [LT, LT],
    Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]],
]:
    def builder(
        pos_label: LT, neg_label: LT
    ) -> Tuple[
        Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]
    ]:
        estimators_a, stops_a = a(pos_label, neg_label)
        estimators_b, stops_b = b(pos_label, neg_label)
        estimators = flatten_dicts(estimators_a, estimators_b)
        stops = flatten_dicts(stops_a, stops_b)
        return estimators, stops

    return builder


def standoff_builder(
    pos_label: LT, neg_label: LT
) -> Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]]:
    recall95 = AprioriRecallTarget(pos_label, 0.95)
    recall100 = AprioriRecallTarget(pos_label, 1.0)
    knee = KneeStoppingRule(pos_label)
    half = ReviewHalfStoppingRule(pos_label)
    budget = BudgetStoppingRule(pos_label)
    rule2399 = Rule2399StoppingRule(pos_label)
    stop200 = StopAfterKNegative(pos_label, 200)
    stop400 = StopAfterKNegative(pos_label, 400)
    criteria = {
        "Perfect95": recall95,
        "Perfect100": recall100,
        "Half": half,
        "Knee": knee,
        "Budget": budget,
        "Rule2399": rule2399,
        "Stop200": stop200,
        "Stop400": stop400,
    }
    return dict(), criteria


TARGETS = [0.7, 0.8, 0.9, 0.95, 1.0]

STOP_BUILDER_REPOSITORY = {
    StopBuilderConfiguration.CHAO_CONS_OPT: conservative_optimistic_builder(
        {"Chao": ChaoEstimator()}, TARGETS
    ),
    StopBuilderConfiguration.CHAO_CONS_OPT_ALT: conservative_optimistic_builder(
        {"ChaoALT": ChaoAlternative()}, TARGETS
    ),
    StopBuilderConfiguration.CHAO_BOTH: combine_builders(
        conservative_optimistic_builder({"Chao": ChaoEstimator()}, TARGETS),
        conservative_optimistic_builder({"ChaoALT": ChaoAlternative()}, TARGETS),
    ),
    StopBuilderConfiguration.AUTOTAR: standoff_builder,
    StopBuilderConfiguration.AUTOSTOP: conservative_optimistic_builder(
        {"AUTOSTOP": HorvitzThompsonVar2()}, TARGETS
    ),
}


EXPERIMENT_REPOSITORY = {
    ExperimentCombination.CHAO: TarExperimentParameters(
        ALConfiguration.ILRaschNBLRRFLGBMRAND,
        None,
        SeparateInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_CONS_OPT,),
        10,
        10,
        10,
    ),
    ExperimentCombination.CHAO_ALT: TarExperimentParameters(
        ALConfiguration.ILRaschNBLRRFLGBMRAND,
        None,
        SeparateInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_CONS_OPT_ALT,),
        10,
        10,
        10,
    ),
    ExperimentCombination.CHAO_BOTH: TarExperimentParameters(
        ALConfiguration.ILRaschNBLRRFLGBMRAND,
        None,
        SeparateInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_BOTH,),
        10,
        10,
        10,
    ),
    ExperimentCombination.AUTOTAR: TarExperimentParameters(
        ALConfiguration.AUTOTAR,
        None,
        RandomInitializer.builder(5),
        (StopBuilderConfiguration.AUTOTAR,),
        10,
        10,
        10,
    ),
    ExperimentCombination.AUTOSTOP: TarExperimentParameters(
        ALConfiguration.AUTOSTOP,
        None,
        RandomInitializer.builder(5),
        (StopBuilderConfiguration.AUTOSTOP,),
        10,
        10,
        10,
    ),
}
