import typing as ty
from abc import ABC, abstractmethod
from collections import OrderedDict
from os import PathLike
from typing import Any, Dict, FrozenSet, Generic, List, Mapping, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..estimation.base import Estimate
from ..typehints import LT
from .experiments import ExperimentIterator
from .statistics import TarDatasetStats, TemporalRecallStats
from .plotter import ExperimentPlotter



class TarExperimentPlotter(ExperimentPlotter[LT], Generic[LT]):
    pos_label: LT
    neg_label: LT

    dataset_stats: ty.OrderedDict[int, TarDatasetStats]
    recall_stats: ty.OrderedDict[int, TemporalRecallStats]
    estimates: ty.OrderedDict[int, Dict[str, Estimate]]
    stop_results: ty.OrderedDict[int, Dict[str, bool]]

    def __init__(self, pos_label: LT, neg_label: LT) -> None:
        self.pos_label = pos_label
        self.neg_label = neg_label
        
        self.dataset_stats = OrderedDict()
        self.recall_stats = OrderedDict()
        self.estimates = OrderedDict()
        self.stop_results = OrderedDict()
        self.it = 0
        self.it_axis: List[int] = list() 

    def update(self, 
               exp_iterator: ExperimentIterator[Any, Any, Any, Any, Any, LT],
               stop_result: Mapping[str, bool]) -> None:
        learner = exp_iterator.learner
        self.it = exp_iterator.it
        self.it_axis.append(self.it)
        self.recall_stats[self.it]  = TemporalRecallStats.from_learner(learner, self.pos_label, self.neg_label)
        self.dataset_stats[self.it] = TarDatasetStats.from_learner(learner, self.pos_label, self.neg_label)
        self.estimates[self.it] = {name: estimate for name, estimate in exp_iterator.recall_estimate.items()}
        self.stop_results[self.it] = stop_result
        
    @property
    def estimator_names(self) -> FrozenSet[str]:
        if self.estimates:
            return frozenset(self.estimates[self.it].keys())
        return frozenset() 
    
    @property
    def criterion_names(self) -> FrozenSet[str]:
        if self.stop_results:
            return frozenset(self.stop_results[self.it].keys())
        return frozenset() 

    def _effort_axis(self) -> np.ndarray:
        effort_axis = np.array([self.recall_stats[it].effort for it in self.it_axis])
        return effort_axis

    def exp_random_recall(self, it: int) -> float:
        effort = self.recall_stats[it].effort
        dataset_size = self.dataset_stats[it].size
        true_pos = self.dataset_stats[it].pos_count
        expected = effort / dataset_size * true_pos
        return expected

    def plot_recall_statistic(self,
                              stats: Mapping[int, TemporalRecallStats], 
                              key: str,                              
                              label: str) -> None:
        effort_axis = self._effort_axis()
        curve = np.array([stats[it].__dict__[key] for it in self.it_axis])
        plt.plot(effort_axis, curve, label=label)

    def plot_estimates(self, key: str, color="gray", alpha=0.2) -> None:
        effort_axis = self._effort_axis()
        points = np.array([self.estimates[it][key].point for it in self.it_axis])
        lows = np.array([self.estimates[it][key].lower_bound for it in self.it_axis])
        uppers = np.array([self.estimates[it][key].upper_bound for it in self.it_axis])
        plt.plot(effort_axis, points, linestyle="-.", label=f"Estimate by {key}")
        plt.fill_between(effort_axis, lows, uppers, color=color, alpha=alpha)

    def plot_stop_criteria(self) -> None:
        for crit_name in self.criterion_names:
            for it in self.it_axis:
                frame = self.stop_results[it]
                if frame[crit_name]:
                    exp_found = self.exp_random_recall(it)
                    act_found = self.recall_stats[it].pos_docs_found
                    wss = self.recall_stats[it].wss
                    recall = self.recall_stats[it].recall
                    plt.vlines(x=self.recall_stats[it].effort,
                               ymin=exp_found, 
                               ymax=act_found,
                               linestyles="dashed",
                               label=f"{crit_name} WSS: {(wss*100):.1f} %, "
                                     f"Recall: {(recall*100):.1f} %")
                    break


    def show(self,
             x_lim: Optional[float] = None,
             y_lim: Optional[float] = None,
             recall_target: float = 0.95,
             filename: "Optional[PathLike[str]]" = None) -> None:
        # Static data
        true_pos = self.dataset_stats[self.it].pos_count
        dataset_size = self.dataset_stats[self.it].size

        pos_target = int(np.ceil(recall_target * true_pos))
        effort_axis =self._effort_axis()
        
        plt.axhline(y=true_pos, linestyle=":", label=f"100 % recall ({true_pos})")
        plt.axhline(y=pos_target, linestyle=":", label=f"{int(recall_target * 100)} % recall ({pos_target})")
        plt.plot(effort_axis, (effort_axis / dataset_size) * true_pos, ":",label = f"Exp. found at random")

        # Recall data
        recall_stats = TemporalRecallStats.transpose_dict(self.recall_stats)
        
        
        # Plot pos docs docs found
        for name, stats in recall_stats.items():
            self.plot_recall_statistic(stats, "pos_docs_found", f"# found by {name}")

        # Plotting estimations
        for estimator in self.estimator_names:
            self.plot_estimates(estimator)

        # Plot stop criteria
        self.plot_stop_criteria()
        
        # Setting axis limitations
        if x_lim is not None:
            plt.xlim(0, x_lim)
        if y_lim is not None:
            plt.ylim(0, y_lim)
        else:
            plt.ylim(0, 1.4 * true_pos)

        # Setting axis labels
        plt.xlabel(f"number of read documents")
        plt.ylabel("number of retrieved relevant documents")
        plt.title(f"Run on a dataset with {int(true_pos)} inclusions out of {int(dataset_size)}")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

    def wss_at_target(self, target: float) -> float:
        for it in self.it_axis:
            frame = self.recall_stats[it]
            if frame.recall >= target:
                return frame.wss
        return float("nan")

    def recall_at_stop(self, stop_criterion: str) -> float:
        for it in self.it_axis:
            frame = self.stop_results[it]
            if frame[stop_criterion]:
                return self.recall_stats[it].recall
        return float("nan")

    def wss_at_stop(self, stop_criterion: str) -> float:
        for it in self.it_axis:
            frame = self.stop_results[it]
            if frame[stop_criterion]:
                return self.recall_stats[it].wss
        return float("nan")
