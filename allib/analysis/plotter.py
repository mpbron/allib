from allib.activelearning.estimator import Estimator
from allib.estimation.abundance import AbundanceEstimator
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, Sequence

import pandas as pd  # type: ignore

from ..activelearning import ActiveLearner
from ..activelearning.ensembles import AbstractEnsemble
from ..estimation.base import AbstractEstimator
from ..utils.func import flatten_dicts

LT = TypeVar("LT")

def name_formatter(learner: ActiveLearner[Any, Any, Any, Any, LT]) -> str:
    name, label = learner.name
    if label is not None:
        return f"{name}_{label}"
    return name

class BinaryPlotter(Generic[LT]):
    result_frame: pd.DataFrame

    def __init__(self, pos_label: LT, neg_label: LT, estimator: AbstractEstimator[Any, Any, Any, Any, LT]):
        self.result_frame = pd.DataFrame()
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.estimator = estimator

    def update(self,
               activelearner: ActiveLearner[Any, Any, Any, Any, LT]
               ) -> None:
        def get_learner_results(learner: ActiveLearner[Any, Any, Any, Any, LT]) -> Dict[str, Optional[Union[int, float]]]:
            name = name_formatter(learner)
            pos_docs = learner.env.labels.get_instances_by_label(
                self.pos_label).intersection(learner.env.labeled)
            neg_docs = learner.env.labels.get_instances_by_label(
                self.neg_label).intersection(learner.env.labeled)
            parent_results = {
                f"{name}_pos_count": len(pos_docs),
                f"{name}_neg_count": len(neg_docs),
                f"{name}_total_count": len(learner.env.labeled)
            }
            if isinstance(learner, AbstractEnsemble):
                child_learners: Sequence[ActiveLearner[Any, Any, Any, Any, LT]] = learner.learners # type: ignore
                child_results = flatten_dicts(*map(get_learner_results, child_learners))
                results = {**parent_results, **child_results}
                return results
            return parent_results # type: ignore     

        name = name_formatter(activelearner)
        count_results = get_learner_results(activelearner)
        estimation_results: Dict[str, Optional[float]] = {}
        if isinstance(self.estimator, AbundanceEstimator):
            assert isinstance(activelearner, Estimator)
            estimations = self.estimator.all_estimations(activelearner, self.pos_label)
            for est_name, estimation, stderr in estimations:
                estimation_results[f"{est_name}_estimation"] = estimation
                estimation_results[f"{est_name}_estimation_error"] = stderr
        
        crit_estimation, crit_error = self.estimator(activelearner, self.pos_label)
        crit_estimation_results = {
                f"{name}_stopcriterion": crit_estimation,
                f"{name}_stopcriterion_error": crit_error,
        }
        
        results = {**count_results, **estimation_results, **crit_estimation_results}
        self.result_frame = self.result_frame.append(results, ignore_index=True)
        pos_count = activelearner.env.labels.document_count(self.pos_label)
        print(f"Found {pos_count} documents so far")


            
            
