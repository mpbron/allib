from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Mapping,
    MutableMapping,
    Sequence,
    Union,
)

import instancelib as il
from instancelib import InstanceProvider, LabelProvider
from instancelib.analysis.base import MulticlassModelMetrics
from instancelib.utils.func import value_map

from ..activelearning.base import ActiveLearner
from ..activelearning.insclass import ILMLBased
from ..estimation.base import AbstractEstimator, Estimate
from ..stopcriterion.base import AbstractStopCriterion
from ..typehints import DT, IT, KT, LT, RT, VT
from .statistics import ALStats, LabelALStatistics


class ExperimentIterator(Generic[IT, KT, DT, VT, RT, LT]):
    learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    it: int
    batch_size: int
    stop_interval: Mapping[str, int]
    stopping_criteria: Mapping[str, AbstractStopCriterion[LT]]
    estimators: Mapping[str, AbstractEstimator[IT, KT, DT, VT, RT, LT]]

    def __init__(
        self,
        learner: ActiveLearner[IT, KT, DT, VT, RT, LT],
        pos_label: LT,
        neg_label: LT,
        stopping_criteria: Mapping[str, AbstractStopCriterion[LT]],
        estimators: Mapping[str, AbstractEstimator[IT, KT, DT, VT, RT, LT]],
        batch_size: int = 1,
        stop_interval: Union[int, Mapping[str, int]] = 1,
        estimation_interval: Union[int, Mapping[str, int]] = 1,
        iteration_hooks: Sequence[
            Callable[[ActiveLearner[IT, KT, DT, VT, RT, LT]], Any]
        ] = list(),
        estimator_hooks: Mapping[
            str, Callable[[AbstractEstimator[IT, KT, DT, VT, RT, LT]], Any]
        ] = dict(),
    ) -> None:
        # Iteration tracker
        self.it = 0
        # Algorithm selection
        self.learner = learner
        self.stopping_criteria = stopping_criteria
        self.estimators = estimators

        # Labels
        self.pos_label = pos_label
        self.neg_label = neg_label

        # Estimation tracker
        self.estimation_tracker: Dict[str, Estimate] = dict()
        self.iteration_hooks = iteration_hooks
        self.estimator_hooks = estimator_hooks

        # Batch sizes
        self.batch_size = batch_size
        self.stop_interval = (
            {k: stop_interval for k in self.stopping_criteria}
            if isinstance(stop_interval, int)
            else stop_interval
        )
        self.estimation_interval = (
            {k: estimation_interval for k in self.estimators}
            if isinstance(estimation_interval, int)
            else estimation_interval
        )

    def _retrain(self) -> None:
        if self.it % self.batch_size == 0:
            self.learner.update_ordering()

    def determine_stop(self) -> Mapping[str, bool]:
        result: Dict[str, bool] = dict()
        for k, crit in self.stopping_criteria.items():
            if self.it % self.stop_interval[k] == 0:
                crit.update(self.learner)
            result[k] = crit.stop_criterion
        return result

    def _estimate_recall(self) -> Mapping[str, Estimate]:
        for k, estimator in self.estimators.items():
            if self.it % self.estimation_interval[k] == 0:
                estimation = estimator(self.learner, self.pos_label)
                self.estimation_tracker[k] = estimation
        return self.estimation_tracker

    @property
    def finished(self) -> bool:
        return self.learner.env.unlabeled.empty

    @property
    def recall_estimate(self) -> Mapping[str, Estimate]:
        return self.estimation_tracker

    def _query_and_label(self) -> None:
        instance = next(self.learner)
        oracle_labels = self.learner.env.truth.get_labels(instance)

        # Set the labels in the active learner
        self.learner.env.labels.set_labels(instance, *oracle_labels)
        self.learner.set_as_labeled(instance)

    def call_hooks(self) -> Sequence[Any]:
        results = [hook(self.learner) for hook in self.iteration_hooks]
        return results

    def call_estimate_hooks(self) -> Mapping[str, Any]:
        results = {
            k: hook(self.estimators[k]) for k, hook in self.estimator_hooks.items()
        }
        return results

    def iter_increment(self) -> None:
        self.it += 1

    def __call__(self) -> Mapping[str, bool]:
        self._retrain()
        self._query_and_label()
        self._estimate_recall()
        stop_result = self.determine_stop()
        self.call_hooks()
        self.iter_increment()
        return stop_result


class ClassificationExperiment(Generic[IT, KT, DT, VT, RT, LT]):
    learners: Mapping[str, ActiveLearner[IT, KT, DT, VT, RT, LT]]
    it: int
    batch_size: int
    stop_interval: Mapping[str, int]
    test_set: InstanceProvider[IT, KT, DT, VT, RT]
    ground_truth: LabelProvider[KT, LT]
    label_stats_tracker: MutableMapping[
        int, Mapping[str, Mapping[LT, LabelALStatistics]]
    ]
    stopping_criteria: Mapping[str, Mapping[str, AbstractStopCriterion[LT]]]
    estimators: Mapping[str, AbstractEstimator[IT, KT, DT, VT, RT, LT]]
    iteration_hooks: Mapping[
        str, Sequence[Callable[[ActiveLearner[IT, KT, DT, VT, RT, LT]], Any]]
    ]
    main_stats_tracker: MutableMapping[int, Mapping[str, ALStats]]
    metrics_tracker: MutableMapping[int, Mapping[str, MulticlassModelMetrics]]
    stop_tracker: MutableMapping[int, Mapping[str, Mapping[str, bool]]]

    def __init__(
        self,
        learners: Mapping[str, ActiveLearner[IT, KT, DT, VT, RT, LT]],
        test_set: InstanceProvider[IT, KT, DT, VT, RT],
        ground_truth: LabelProvider[KT, LT],
        stopping_criteria: Mapping[
            str, Mapping[str, AbstractStopCriterion[LT]]
        ] = dict(),
        batch_size: int = 1,
        stop_interval: Union[int, Mapping[str, int]] = 1,
        iteration_hooks: Mapping[
            str, Sequence[Callable[[ActiveLearner[IT, KT, DT, VT, RT, LT]], Any]]
        ] = dict(),
    ) -> None:
        # Iteration tracker
        self.it = 0
        # Algorithm selection
        self.learners = learners
        self.test_set = test_set
        self.ground_truth = ground_truth
        self.stopping_criteria = stopping_criteria

        self.iteration_hooks = iteration_hooks

        self.label_stats_tracker = dict()
        self.metrics_tracker = dict()
        self.stop_tracker = dict()
        self.main_stats_tracker = dict()

        # Batch sizes
        self.batch_size = batch_size
        self.stop_interval = (
            {k: stop_interval for k in self.stopping_criteria}
            if isinstance(stop_interval, int)
            else stop_interval
        )

    def _retrain(self) -> None:
        if self.it % self.batch_size == 0:
            for learner in self.learners.values():
                learner.update_ordering()

    def determine_stop(self) -> Mapping[str, Mapping[str, bool]]:
        result: Dict[str, Dict[str, bool]] = dict()
        for l_key, learner in self.learners.items():
            if self.stopping_criteria and l_key in self.stopping_criteria:
                for k, crit in self.stopping_criteria[l_key].items():
                    if self.it % self.stop_interval[k] == 0:
                        crit.update(learner)
                    result[l_key][k] = crit.stop_criterion
        return result

    def _label_stats(self) -> Mapping[str, Mapping[LT, LabelALStatistics[LT]]]:
        result: Dict[str, Dict[LT, LabelALStatistics]] = dict()
        for l_key, learner in self.learners.items():
            for label in learner.env.labels.labelset:
                subset = learner.env.get_subset_by_labels(learner.env.labeled, label)
                generated = frozenset(
                    learner.env.get_subset_by_labels(learner.env.all_instances, label)
                ).difference(subset)
                stats = LabelALStatistics(label, len(subset), len(generated))
                result.setdefault(l_key, dict())[label] = stats
        return result

    def _al_stats(self) -> Mapping[str, ALStats]:
        result = {
            l_key: ALStats(
                len(learner.env.unlabeled),
                len(learner.env.labeled),
                len(learner.env.dataset),
            )
            for l_key, learner in self.learners.items()
        }
        return result

    @property
    def label_stats(self) -> Mapping[str, Mapping[LT, LabelALStatistics]]:
        return self.label_stats_tracker[self.it - 1]

    @property
    def stop_result(self) -> Mapping[str, Mapping[str, bool]]:
        return self.stop_tracker[self.it - 1]

    def _calculate_metrics(self) -> Mapping[str, MulticlassModelMetrics]:
        result: Dict[str, MulticlassModelMetrics] = dict()
        for l_key, learner in self.learners.items():
            if isinstance(learner, ILMLBased):
                result[l_key] = il.classifier_performance(
                    learner.classifier, self.test_set, self.ground_truth
                )
        return result

    @property
    def finished(self) -> bool:
        return all((learner.env.unlabeled.empty for learner in self.learners.values()))

    @property
    def metrics(self) -> Mapping[str, MulticlassModelMetrics]:
        return self.metrics_tracker[self.it - 1]

    @property
    def main_stats(self) -> Mapping[str, ALStats]:
        return self.main_stats_tracker[self.it - 1]

    def _query_and_label(self) -> None:
        for learner in self.learners.values():
            instance = next(learner)
            oracle_labels = learner.env.truth.get_labels(instance)

            # Set the labels in the active learner
            learner.env.labels.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)

    def call_hooks(self) -> Mapping[str, Sequence[Any]]:
        results = {
            l_key: [hook(self.learners[l_key]) for hook in hooks]
            for l_key, hooks in self.iteration_hooks.items()
        }
        return results

    def iter_increment(self) -> None:
        self.it += 1

    def __call__(self) -> Mapping[str, Mapping[str, bool]]:
        self._retrain()
        self._query_and_label()
        self.main_stats_tracker[self.it] = self._al_stats()
        self.label_stats_tracker[self.it] = self._label_stats()
        self.metrics_tracker[self.it] = self._calculate_metrics()
        stop_result = self.determine_stop()
        self.stop_tracker[self.it] = stop_result
        self.call_hooks()
        self.iter_increment()
        return stop_result
