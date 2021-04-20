import functools
from os import PathLike
from typing import (Any, Dict, Generic, Optional, Sequence, Tuple, TypeVar,
                    Union)

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..activelearning import ActiveLearner
from ..activelearning.ensembles import AbstractEnsemble
from ..activelearning.estimator import Estimator
from ..estimation.base import AbstractEstimator
from ..estimation.rcapture import AbundanceEstimator
from ..utils.func import flatten_dicts
from .analysis import process_performance

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
        def get_learner_results(learner: ActiveLearner[Any, Any, Any, Any, LT], root = True) -> Dict[str, Optional[Union[int, float]]]:
            name = name_formatter(learner)
            pos_docs = learner.env.labels.get_instances_by_label(
                self.pos_label).intersection(learner.env.labeled)
            neg_docs = learner.env.labels.get_instances_by_label(
                self.neg_label).intersection(learner.env.labeled)
            if root:
                parent_results = {
                    f"positives": len(pos_docs),
                    f"negatives": len(neg_docs),
                    f"total": len(learner.env.labeled)
                }
            else:
                parent_results = {
                    f"{name}_pos_count": len(pos_docs),
                    f"{name}_neg_count": len(neg_docs),
                    f"{name}_total_count": len(learner.env.labeled)
                }
            if isinstance(learner, AbstractEnsemble):
                child_learners: Sequence[ActiveLearner[Any, Any, Any, Any, LT]] = learner.learners # type: ignore
                child_func = functools.partial(get_learner_results, root=False)
                child_results = flatten_dicts(*map(child_func, child_learners))
                results = {**parent_results, **child_results}
                return results
            return parent_results # type: ignore
        
        performance = process_performance(activelearner, self.pos_label)
        stats = {
            "true_pos_count" : activelearner.env.truth.document_count(self.pos_label),
            "true_neg_count" : activelearner.env.truth.document_count(self.neg_label),
            "dataset_size": len(activelearner.env.dataset),
            "wss": performance.wss,
            "recall": performance.recall
        }

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
        results = {**count_results, **estimation_results, **crit_estimation_results, **stats}
        self.result_frame = self.result_frame.append(results, ignore_index=True)
        pos_count = activelearner.env.labels.document_count(self.pos_label)
        print(f"Found {pos_count} documents so far")
    
    def show(self,
             x_lim: Optional[float] = None,
             y_lim: Optional[float] = None,
             all_estimations: bool = False,
             filename: Optional[PathLike] = None) -> None:
        # Gathering intermediate results
        df = self.result_frame

        n_pos_overall = np.array(df.positives)
        n_found = int(n_pos_overall[-1])
        true_pos = int(np.array(df.true_pos_count)[-1])
        n_read = int(np.array(df.total)[-1])
        total_n = int(np.array(df.dataset_size)[-1])
        wss = np.array(df.wss)[-1] * 100
        recall = np.array(df.recall)[-1] * 100
        n_exp = int(np.floor((n_read / total_n) * true_pos))

        # Use n_documents as the x-axis
        n_documents_col = "total"
        n_documents_read = np.array(df[n_documents_col]) # type: ignore
        plt.plot(n_documents_read, n_pos_overall, label=f"# total found ({n_found})")
        
        # Plotting positive document counts
        pos_counts = df.filter(regex="pos_count$") # type: ignore
        n95 = int(np.ceil(0.95 * true_pos))
        pos_cols = filter(lambda c: c != "true_pos_count", pos_counts.columns)
        for i, col in enumerate(pos_cols , 1):
            total_learner = int(pos_counts[col].iloc[-1]) # type: ignore
            plt.plot(n_documents_read, 
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
        if x_lim is not None:
            plt.xlim(0, x_lim)
        if y_lim is not None:
            plt.ylim(0, y_lim)
        else:
            plt.ylim(0, 1.4 * true_pos)

        # Setting axis labels
        plt.xlabel(f"number of read documents (total {n_read}), WSS = {wss:.2f} % Recall = {recall:.2f} %")
        plt.ylabel("number of documents")
        plt.title(f"Run on a dataset with {int(true_pos)} inclusions out of {int(total_n)}")
        if not all_estimations:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')


            
            
