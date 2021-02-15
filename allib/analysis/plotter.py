from allib.activelearning.estimator import Estimator
from allib.estimation.rcapture import AbundanceEstimator
from typing import Any, Dict, Generic, Optional, Tuple, TypeVar, Union, Sequence

import pandas as pd  # type: ignore
import numpy as np # type: ignore
import matplotlib.pyplot as plt  # type: ignore

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

    def __init__(self, pos_label: LT, neg_label: LT, *estimators: AbstractEstimator[Any, Any, Any, Any, LT]):
        self.result_frame = pd.DataFrame()
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.estimators = list(estimators)

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
        # if isinstance(self.estimator, AbundanceEstimator):
        #     assert isinstance(activelearner, Estimator)
        #     estimations = self.estimator.all_estimations(activelearner, self.pos_label)
        #     for est_name, estimation, stderr in estimations:
        #         estimation_results[f"{est_name}_estimation"] = estimation
        #         estimation_results[f"{est_name}_estimation_error"] = stderr
        crit_estimation_results: Dict[str, Optional[float]] = dict()
        for estimator in self.estimators:
            crit_estimation, crit_lowerbound, crit_upperbound = estimator(activelearner, self.pos_label)
            crit_estimation_results[f"{name}_{estimator.name}_stopcriterion"] = crit_estimation
            crit_estimation_results[f"{name}_{estimator.name}_stopcriterion_low"] = crit_lowerbound
            crit_estimation_results[f"{name}_{estimator.name}_stopcriterion_up"] = crit_upperbound
        results = {**count_results, **estimation_results, **crit_estimation_results}
        self.result_frame = self.result_frame.append(results, ignore_index=True)
        pos_count = activelearner.env.labels.document_count(self.pos_label)
        print(f"Found {pos_count} documents so far")
    
    def show(self,
             learner: ActiveLearner[Any, Any, Any, Any, LT],
             all_estimations: bool,
             y_lim_scale: float = 1.4) -> None:
        n_found = learner.env.labels.document_count(self.pos_label)
        true_pos = learner.env.truth.document_count(self.pos_label)
        n_read = len(learner.env.labeled)
        total_n = len(learner.env.dataset)
        n_exp = int(np.floor((n_read / total_n) * true_pos))
        
        # Gathering intermediate results
        df = self.result_frame

        # Use n_documents as the x-axis
        n_documents_col = f"{name_formatter(learner)}_total_count"
        n_documents_read = np.array(df[n_documents_col]) # type: ignore

        # Plotting positive document counts
        pos_counts = df.filter(regex="pos_count$") # type: ignore
        n95 = int(np.ceil(0.95 * true_pos))
        for i, col in enumerate(pos_counts.columns):
            if i == 0:
                plt.plot(n_documents_read, pos_counts[col], label=f"# total found ({n_found})")
            else:
                total_learner = int(pos_counts[col].iloc[-1])
                plt.plot(
                    n_documents_read, 
                    pos_counts[col], 
                    label=f"# found by $L_{i}$ ({total_learner})")
        if all_estimations:
            # Plotting estimations
            estimations = df.filter(regex="estimation$") #type: ignore
            for col in estimations.columns:
                ests = estimations[col]
                error_colname = f"{col}_error"
                errors = df[error_colname]
                plt.plot(n_documents_read, ests, "-.", label="Estimate") # type: ignore
                plt.fill_between(n_documents_read, # type: ignore
                                 ests - errors, # Lower bound
                                 ests + errors, # Upper bound
                                 color='gray', alpha=0.2)

        # Plotting estimations for main stop criterion
        stopcriterion = df.filter(regex="stopcriterion$") # type: ignore
        for i, col in enumerate(stopcriterion.columns):
            ests = stopcriterion[col]
            low_colname = f"{col}_low"
            upper_colname = f"{col}_up"
            lower_bounds = np.array(df[low_colname])
            upper_bounds = np.array(df[upper_colname])
            plt.plot(n_documents_read, ests, "-.", label=f"Estimate ({i})") # type: ignore
            plt.fill_between(n_documents_read, # type: ignore
                             lower_bounds, # type: ignore
                             upper_bounds, # type: ignore
                             color='gray', alpha=0.2)
            print(f"{i}: {col}")
        
        # Plotting target boundaries 
        plt.plot(n_documents_read, ([true_pos] *len(n_documents_read)), ":", label=f"100 % recall ({true_pos})")
        plt.plot(n_documents_read, ([n95] *len(n_documents_read)), ":", label=f"95 % recall ({n95})")
        
        # Plotting expected random documents found
        plt.plot(n_documents_read, (n_documents_read / total_n) * true_pos, ":",label = f"Exp. found at random ({n_exp})")
        
        # Setting axis limitations
        plt.ylim(0, y_lim_scale * true_pos)

        # Setting axis labels
        plt.xlabel(f"number of read documents (total {n_read})")
        plt.ylabel("number of documents")
        if not all_estimations:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')


            
            
