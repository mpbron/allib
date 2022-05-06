from typing import Any, Callable, Dict, Generic, Mapping, Optional, Sequence, Union

from ..activelearning.base import ActiveLearner
from ..estimation.base import AbstractEstimator, Estimate
from ..stopcriterion.base import AbstractStopCriterion
from ..typehints import DT, IT, KT, LT, RT, VT


class ExperimentIterator(Generic[IT, KT, DT, VT, RT, LT]):
    learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    it: int
    batch_size: int
    stop_interval: Mapping[str, int]
    stopping_criteria: Mapping[str, AbstractStopCriterion[LT]]
    estimators: Mapping[str, AbstractEstimator[IT, KT, DT, VT, RT, LT]]

    def __init__(self, 
                 learner: ActiveLearner[IT, KT, DT, VT, RT ,LT],
                 pos_label: LT,
                 neg_label: LT,
                 stopping_criteria: Mapping[str,AbstractStopCriterion[LT]],
                 estimators: Mapping[str, AbstractEstimator[IT, KT, DT, VT, RT, LT]],
                 batch_size: int = 1, 
                 stop_interval: Union[int, Mapping[str, int]] = 1,
                 estimation_interval: Union[int, Mapping[str, int]] = 1,
                 iteration_hooks: Sequence[Callable[[ActiveLearner[IT, KT, DT, VT, RT ,LT]], Any]] = list(),
                 estimator_hooks: Mapping[str, Callable[[AbstractEstimator[IT, KT, DT, VT, RT, LT]], Any]] = dict()) -> None:
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
        self.estimation_tracker: Dict[str, Estimate]= dict()
        self.iteration_hooks = iteration_hooks
        self.estimator_hooks = estimator_hooks

        # Batch sizes
        self.batch_size = batch_size
        self.stop_interval = {k: stop_interval for k in self.stopping_criteria} if isinstance(stop_interval, int) else stop_interval 
        self.estimation_interval = {k: estimation_interval for k in self.estimators} if isinstance(estimation_interval, int) else estimation_interval

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
        return self.learner.env.labeled.empty

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

    def call_estimate_hooks(self) -> Mapping[str,Any]:
        results = {k: hook(self.estimators[k]) for k,hook in self.estimator_hooks.items()}
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


