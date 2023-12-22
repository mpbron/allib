from dataclasses import dataclass
from typing import Any, Callable, Generic, Mapping, Optional, Sequence, Tuple, TypeVar

import numpy.typing as npt
from instancelib.utils.func import flatten_dicts, list_unzip

from ..estimation.quant import QuantEstimator

from ..analysis.initialization import (
    AutoStopLargeInitializer,
    CMHInitializer,
    Initializer,
    PriorInitializer,
    RandomInitializer,
    SeededEnsembleInitializer,
    SeededRandomInitializer,
    SeparateInitializer,
    TargetInitializer,
)
from ..estimation.autostop import HorvitzThompsonVar2, HorvitzThompsonVar1
from ..estimation.base import AbstractEstimator
from ..estimation.catalog import EstimatorCatalog
from ..estimation.mhmodel import (
    ChaoAlternative,
    BiasFallback,
    BiasCorrectedEstimator,
    ChaoRivestEstimator,
    Chao1989Estimator,
    Chao1987Estimator,
    ChaoRivestOriginalPoint,
    NMinWhenAvailable,
    LogLinear,
    OnlyNMin,
    BiasFallbackVariance,
    Chao2006,
)
from ..estimation.rasch_comb_parametric import EMRaschRidgeParametricPython
from ..estimation.rasch_multiple import EMRaschRidgeParametricConvPython
from ..estimation.rasch_parametric import ParametricRaschPython
from ..estimation.rasch_python import EMRaschRidgePython
from ..module import ModuleCatalog as Cat
from ..stopcriterion.base import AbstractStopCriterion
from ..stopcriterion.catalog import StopCriterionCatalog
from ..stopcriterion.estimation import Conservative, Optimistic
from ..stopcriterion.heuristic import AprioriRecallTarget
from ..stopcriterion.others import (
    BudgetStoppingRule,
    CMH_HeuristicStoppingRule,
    KneeStoppingRule,
    QuantStoppingRule,
    ReviewHalfStoppingRule,
    Rule2399StoppingRule,
    StopAfterKNegative,
)
from ..stopcriterion.sequence import LastSequence, PriorToLast
from ..stopcriterion.target import TargetCriterion
from ..typehints import LT
from .catalog import (
    ALConfiguration,
    EstimationConfiguration,
    ExperimentCombination,
    FEConfiguration,
    StopBuilderConfiguration,
)
from .ensemble import (
    LGBM,
    al_config_ensemble_prob,
    al_config_entropy,
    autotar_ensemble,
    chao_ensemble,
    chao_ensemble2,
    chao_ensemble_same,
    chao_ensemble_prior,
    chao_ensemble_same_random,
    naive_bayes_estimator,
    rasch_estimator,
    rasch_lr,
    rasch_nblrrf,
    rasch_nblrrflgbm,
    rasch_nblrrflgbmrand,
    rasch_nblrrfsvm,
    rasch_rf,
    svm_estimator,
    targetmethod,
    tf_idf5000,
    cmhmethod,
)
from .tarbaselines import autostop, autostop_large, autotar
from ..stopcriterion.cmh import CMH_CertificationRule, CMH_HeuristicMethodRuleTwoPhase

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
    ALConfiguration.CHAO_ENSEMBLE: chao_ensemble(1),
    ALConfiguration.AUTOTAR: autotar,
    ALConfiguration.AUTOSTOP: autostop,
    ALConfiguration.CHAO_AT_ENSEMBLE: autotar_ensemble,
    ALConfiguration.CHAO_IB_ENSEMBLE: chao_ensemble(
        1, method=Cat.AL.CustomMethods.INCREASING_BATCH
    ),
    ALConfiguration.CHAO_IB_ENSEMBLE_20: chao_ensemble(
        20, method=Cat.AL.CustomMethods.INCREASING_BATCH
    ),
    ALConfiguration.CHAO_SAME: chao_ensemble_same(
        1, method=Cat.AL.CustomMethods.INCREASING_BATCH, clf=LGBM
    ),
    ALConfiguration.TARGET: targetmethod(),
    ALConfiguration.PRIOR: chao_ensemble_prior(
        1, method=Cat.AL.CustomMethods.INCREASING_BATCH, nneg=50, nirel=10
    ),
    ALConfiguration.AUTOSTOP_LARGE_CONS_95: autostop_large(
        StopCriterionCatalog.CONSERVATIVE, EstimatorCatalog.HorvitzThompson2, 0.95
    ),
    ALConfiguration.AUTOSTOP_LARGE_CONS_100: autostop_large(
        StopCriterionCatalog.CONSERVATIVE, EstimatorCatalog.HorvitzThompson2, 1.0
    ),
    ALConfiguration.AUTOSTOP_LARGE_OPT_95: autostop_large(
        StopCriterionCatalog.OPTIMISTIC, EstimatorCatalog.HorvitzThompson2, 0.95
    ),
    ALConfiguration.AUTOSTOP_LARGE_OPT_100: autostop_large(
        StopCriterionCatalog.OPTIMISTIC, EstimatorCatalog.HorvitzThompson2, 1.0
    ),
    ALConfiguration.CMH: cmhmethod(),
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
    EstimationConfiguration.CHAO: ChaoRivestEstimator[Any, Any, Any, Any, Any, str](),
    EstimationConfiguration.AUTOSTOP: HorvitzThompsonVar1(),
    EstimationConfiguration.LOGLINEAR: LogLinear[Any, Any, Any, Any, Any, str](),
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


def combine_builders_long(
    *builders: Callable[
        [LT, LT],
        Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]],
    ]
) -> Callable[
    [LT, LT],
    Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]],
]:
    def builder(
        pos_label: LT, neg_label: LT
    ) -> Tuple[
        Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]
    ]:
        estimators, criteria = list_unzip([bf(pos_label, neg_label) for bf in builders])
        estimator_dict = flatten_dicts(*estimators)
        criteria_dict = flatten_dicts(*criteria)
        return estimator_dict, criteria_dict

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


TARGETS = [0.7, 0.8, 0.9, 0.95, 1.0]


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
    cmh = {
        f"CMH_Standoff_{t}": CMH_HeuristicStoppingRule(pos_label, t, alpha=0.05) for t in TARGETS
    }
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
    return dict(), {**criteria, **cmh}


def target_builder(
    pos_label: LT, neg_label: LT
) -> Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]]:
    return dict(), {"TARGET": TargetCriterion(pos_label)}


def cmhs(
    pos_label: LT,
    neg_label: LT,
    recall_targets: Sequence[float] = (0.95,),
    alpha: float = 0.05,
) -> Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]]:
    cmh_certs = {
        f"CMH_Hybrid_{target}": CMH_CertificationRule(pos_label, target, alpha)
        for target in recall_targets
    }
    cmh_heurs = {
        f"CMH_Standoff_{target}": CMH_HeuristicMethodRuleTwoPhase(pos_label, target, alpha)
        for target in recall_targets
    }
    return dict(), {**cmh_certs, **cmh_heurs}


def last_seq_builder(
    pos_label: LT, neg_label: LT
) -> Tuple[Mapping[str, AbstractEstimator], Mapping[str, AbstractStopCriterion[LT]]]:
    return dict(), {"AUTOSTOP": PriorToLast()}


STOP_BUILDER_REPOSITORY = {
    StopBuilderConfiguration.CHAO_CONS_OPT: combine_builders_long(
        conservative_optimistic_builder(
            {"Chao(Rivest)": ChaoRivestEstimator()}, TARGETS
        ),
        conservative_optimistic_builder(
            {"Chao(1987)NMIN": NMinWhenAvailable()}, TARGETS
        ),
        conservative_optimistic_builder({"Chao(1987)": BiasFallback()}, TARGETS),
        conservative_optimistic_builder(
            {"Chao(1987)": BiasFallbackVariance()}, TARGETS
        ),
        conservative_optimistic_builder(
            {"Chao(1987)STRICT": Chao1987Estimator()}, TARGETS
        ),
        conservative_optimistic_builder(
            {"Chao(2006)": BiasCorrectedEstimator()}, TARGETS
        ),
        conservative_optimistic_builder({"Chao(1989)": Chao1989Estimator()}, TARGETS),
        conservative_optimistic_builder({"Chao(1987)NMINONLY": OnlyNMin()}, TARGETS),
    ),
    StopBuilderConfiguration.QUANT: conservative_optimistic_builder(
        {"Quant": QuantEstimator(2.0)}, TARGETS
    ),
    StopBuilderConfiguration.CHAO_CONS_OPT_ALT: combine_builders_long(
        conservative_optimistic_builder(
            {"Chao(1987)V": BiasFallbackVariance()}, TARGETS
        ),
        conservative_optimistic_builder(
            {"Chao(2006)V": Chao2006()}, TARGETS
        )),
    StopBuilderConfiguration.CHAO_BOTH: combine_builders_long(
        conservative_optimistic_builder(
            {"Chao(1987)": BiasFallbackVariance()}, TARGETS
        ),
        conservative_optimistic_builder(
            {"Chao(2006)": Chao2006()}, TARGETS
        ),
        conservative_optimistic_builder({"Chao(Rivest)": ChaoRivestEstimator()}, TARGETS),
        conservative_optimistic_builder({"Chao(Rivest)O": ChaoRivestOriginalPoint()}, TARGETS),
        conservative_optimistic_builder({"Chao(1987)OLD": BiasFallback()}, TARGETS),
    ),
    StopBuilderConfiguration.RCAPTURE_ALL: combine_builders(
        conservative_optimistic_builder({"Chao": ChaoRivestEstimator()}, TARGETS),
        conservative_optimistic_builder({"LL": LogLinear()}, TARGETS),
    ),
    StopBuilderConfiguration.AUTOTAR: standoff_builder,
    StopBuilderConfiguration.AUTOSTOP: conservative_optimistic_builder(
        {"AUTOSTOP": HorvitzThompsonVar2()}, TARGETS
    ),
    StopBuilderConfiguration.TARGET: target_builder,
    StopBuilderConfiguration.LASTSEQUENCE: last_seq_builder,
    StopBuilderConfiguration.CMH: cmhs,
}


EXPERIMENT_REPOSITORY: Mapping[ExperimentCombination, TarExperimentParameters] = {
    ExperimentCombination.CHAO: TarExperimentParameters(
        ALConfiguration.CHAO_IB_ENSEMBLE,
        None,
        SeededEnsembleInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_BOTH,),
        10,
        10,
        10,
    ),
    ExperimentCombination.CHAO_ALT: TarExperimentParameters(
        ALConfiguration.CHAO_ENSEMBLE,
        None,
        SeededEnsembleInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_CONS_OPT_ALT, StopBuilderConfiguration.AUTOTAR),
        10,
        10,
        10,
    ),
    ExperimentCombination.CHAO_BOTH: TarExperimentParameters(
        ALConfiguration.CHAO_ENSEMBLE,
        None,
        SeededEnsembleInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_BOTH, StopBuilderConfiguration.AUTOTAR),
        10,
        10,
        10,
    ),
    ExperimentCombination.AUTOTAR: TarExperimentParameters(
        ALConfiguration.AUTOTAR,
        None,
        SeededRandomInitializer.builder(5),
        (StopBuilderConfiguration.AUTOTAR,StopBuilderConfiguration.QUANT,),
        10,
        10,
        10,
    ),
    ExperimentCombination.AUTOTAR_QUANT: TarExperimentParameters(
        ALConfiguration.AUTOTAR,
        None,
        SeededRandomInitializer.builder(5),
        (StopBuilderConfiguration.QUANT,),
        10,
        10,
        10,
    ),
    ExperimentCombination.AUTOSTOP: TarExperimentParameters(
        ALConfiguration.AUTOSTOP,
        None,
        SeededRandomInitializer.builder(5),
        (StopBuilderConfiguration.AUTOSTOP, StopBuilderConfiguration.AUTOTAR),
        10,
        10,
        10,
    ),
    ExperimentCombination.CHAO_AT: TarExperimentParameters(
        ALConfiguration.CHAO_AT_ENSEMBLE,
        None,
        SeededEnsembleInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_BOTH, StopBuilderConfiguration.AUTOTAR),
        10,
        10,
        10,
    ),
    ExperimentCombination.CHAO_IB: TarExperimentParameters(
        ALConfiguration.CHAO_IB_ENSEMBLE,
        None,
        SeededEnsembleInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_CONS_OPT, StopBuilderConfiguration.AUTOTAR),
        10,
        10,
        10,
    ),
    ExperimentCombination.CHAO_IB_20: TarExperimentParameters(
        ALConfiguration.CHAO_IB_ENSEMBLE_20,
        None,
        SeededEnsembleInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_CONS_OPT, StopBuilderConfiguration.AUTOTAR),
        10,
        10,
        10,
    ),
    ExperimentCombination.CHAO_SAME: TarExperimentParameters(
        ALConfiguration.CHAO_SAME,
        None,
        SeededEnsembleInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_CONS_OPT, StopBuilderConfiguration.AUTOTAR),
        10,
        10,
        10,
    ),
    ExperimentCombination.RCAPTURE: TarExperimentParameters(
        ALConfiguration.CHAO_IB_ENSEMBLE,
        None,
        SeededEnsembleInitializer.builder(1),
        (StopBuilderConfiguration.RCAPTURE_ALL, StopBuilderConfiguration.AUTOTAR),
        10,
        10,
        10,
    ),
    ExperimentCombination.PRIOR: TarExperimentParameters(
        ALConfiguration.PRIOR,
        None,
        PriorInitializer.builder(1),
        (StopBuilderConfiguration.CHAO_CONS_OPT, StopBuilderConfiguration.AUTOTAR),
        10,
        10,
        10,
    ),
    ExperimentCombination.TARGET: TarExperimentParameters(
        ALConfiguration.TARGET,
        None,
        TargetInitializer.builder(1),
        (StopBuilderConfiguration.TARGET,),
        10,
        10,
        10,
    ),
    ExperimentCombination.AUTOSTOP_LARGE_CONS_95: TarExperimentParameters(
        ALConfiguration.AUTOSTOP_LARGE_CONS_95,
        None,
        AutoStopLargeInitializer.builder(5, None),
        (StopBuilderConfiguration.LASTSEQUENCE,),
        10,
        10,
        10,
    ),
    ExperimentCombination.AUTOSTOP_LARGE_CONS_100: TarExperimentParameters(
        ALConfiguration.AUTOSTOP_LARGE_CONS_100,
        None,
        AutoStopLargeInitializer.builder(5, None),
        (StopBuilderConfiguration.LASTSEQUENCE,),
        10,
        10,
        10,
    ),
    ExperimentCombination.AUTOSTOP_LARGE_OPT_95: TarExperimentParameters(
        ALConfiguration.AUTOSTOP_LARGE_OPT_95,
        None,
        AutoStopLargeInitializer.builder(5, None),
        (StopBuilderConfiguration.LASTSEQUENCE,),
        10,
        10,
        10,
    ),
    ExperimentCombination.AUTOSTOP_LARGE_OPT_100: TarExperimentParameters(
        ALConfiguration.AUTOSTOP_LARGE_OPT_100,
        None,
        AutoStopLargeInitializer.builder(5, None),
        (StopBuilderConfiguration.LASTSEQUENCE,),
        10,
        10,
        10,
    ),
    ExperimentCombination.CMH: TarExperimentParameters(
        ALConfiguration.CMH,
        None,
        CMHInitializer.builder(5),
        (StopBuilderConfiguration.CMH,),
        10,
        10,
        10,
    ),
}
