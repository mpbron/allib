import typing as ty
from abc import ABC, abstractmethod
from collections import OrderedDict
from os import PathLike
from typing import (Any, Dict, FrozenSet, Generic, Iterable, List, Mapping,
                    Optional, Sequence, Tuple, TypeVar, Union)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ..analysis.statistics import EstimationModelStatistics
from ..estimation.base import AbstractEstimator, Estimate
from ..typehints import LT
from .experiments import ExperimentIterator
from .plotter import ExperimentPlotter
from .statistics import TarDatasetStats, TemporalRecallStats

_T = TypeVar("_T")
_V = TypeVar("_V")
_W = TypeVar("_W")


def smooth_similar(
    xs: Union[npt.NDArray[Any], Sequence[_T]], ys: Union[npt.NDArray[Any], Sequence[_V]]
) -> Tuple[Sequence[_T], Sequence[_V]]:
    assert len(xs) == len(ys), f"len(xs) != len(ys); {len(xs)} !=  {len(ys)}"
    x_smoothed: List[_T] = list()
    y_smoothed: List[_V] = list()
    previous_y: Optional[_V] = None
    for x, y in zip(xs, ys):
        if previous_y != y:
            x_smoothed.append(x)
            y_smoothed.append(y)
            previous_y = y
    return x_smoothed, y_smoothed


def smooth_similar3(
    xs: Union[npt.NDArray[Any], Sequence[_T]],
    ys: Union[npt.NDArray[Any], Sequence[_V]],
    zs: Union[npt.NDArray[Any], Sequence[_W]],
) -> Tuple[Sequence[_T], Sequence[_V], Sequence[_W]]:
    assert (
        len(xs) == len(ys) == len(zs)
    ), f"len(xs) != len(ys) != len(zs); {len(xs)} !=  {len(ys)} != {len(zs)}"
    x_smoothed: List[_T] = list()
    y_smoothed: List[_V] = list()
    z_smoothed: List[_W] = list()
    previous_y: Optional[_V] = None
    previous_z: Optional[_W] = None
    for x, y, z in zip(xs, ys, zs):
        if previous_y != y or previous_z != z:
            x_smoothed.append(x)
            y_smoothed.append(y)
            z_smoothed.append(z)
            previous_y = y
            previous_z = z
    return x_smoothed, y_smoothed, z_smoothed


class TarExperimentPlotter(ExperimentPlotter[LT], Generic[LT]):
    pos_label: LT
    neg_label: LT

    dataset_name: str

    dataset_stats: ty.OrderedDict[int, TarDatasetStats]
    recall_stats: ty.OrderedDict[int, TemporalRecallStats]
    estimates: ty.OrderedDict[int, Mapping[str, Estimate]]
    stop_results: ty.OrderedDict[int, Mapping[str, bool]]

    def __init__(self, pos_label: LT, neg_label: LT, dataset_name: str = "") -> None:
        self.pos_label = pos_label
        self.neg_label = neg_label

        self.dataset_name = dataset_name

        self.dataset_stats = OrderedDict()
        self.recall_stats = OrderedDict()
        self.estimates = OrderedDict()
        self.stop_results = OrderedDict()
        self.it = 0
        self.it_axis: List[int] = list()

    def update(
        self,
        exp_iterator: ExperimentIterator[Any, Any, Any, Any, Any, LT],
        stop_result: Mapping[str, bool],
    ) -> None:
        learner = exp_iterator.learner
        self.it = exp_iterator.it
        self.it_axis.append(self.it)
        self.recall_stats[self.it] = TemporalRecallStats.from_learner(
            learner, self.pos_label, self.neg_label
        )
        self.dataset_stats[self.it] = TarDatasetStats.from_learner(
            learner, self.pos_label, self.neg_label
        )
        self.estimates[self.it] = {
            name: estimate for name, estimate in exp_iterator.recall_estimate.items()
        }
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

    def _effort_axis(self) -> npt.NDArray[Any]:
        effort_axis = np.array([self.recall_stats[it].effort for it in self.it_axis])
        return effort_axis

    def exp_random_recall(self, it: int) -> float:
        effort = self.recall_stats[it].effort
        dataset_size = self.dataset_stats[it].size
        true_pos = self.dataset_stats[it].pos_count
        expected = effort / dataset_size * true_pos
        return expected

    def print_last_stats(self) -> None:
        estimate = self.estimates[self.it]
        recall = self.recall_stats[self.it]
        print(estimate)
        print(recall.pos_docs_found)

    def plot_recall_statistic(
        self, stats: Mapping[int, TemporalRecallStats], key: str, label: str
    ) -> None:
        effort_axis = self._effort_axis()
        curve = np.array([stats[it].__dict__[key] for it in self.it_axis])
        plt.plot(effort_axis, curve, label=label)

    def _plot_estimator(self, key: str, color="gray", alpha=0.2) -> None:
        effort_axis = self._effort_axis()
        points = np.array([self.estimates[it][key].point for it in self.it_axis])
        lows = np.array([self.estimates[it][key].lower_bound for it in self.it_axis])
        uppers = np.array([self.estimates[it][key].upper_bound for it in self.it_axis])
        xs, ys = smooth_similar(effort_axis, points)
        xrs, ls, us = smooth_similar3(effort_axis, lows, uppers)
        plt.plot(xs, ys, linestyle="-.", label=f"Estimate by {key}", color=color)
        plt.fill_between(xrs, ls, us, color=color, alpha=alpha)  # type: ignore

    def _plot_stop_criteria(self) -> None:
        results: Sequence[Tuple[int, float, float, str, str]] = list()
        for i, crit_name in enumerate(self.criterion_names):
            color = f"C{i}"
            for it in self.it_axis:
                frame = self.stop_results[it]
                if frame[crit_name]:
                    wss = self.recall_stats[it].wss
                    recall = self.recall_stats[it].recall
                    results.append((it, recall, wss, crit_name, color))
                    break
        results_sorted = sorted(results)
        for it, recall, wss, crit_name, color in results_sorted:
            exp_found = self.exp_random_recall(it)
            act_found = self.recall_stats[it].pos_docs_found
            plt.vlines(
                x=self.recall_stats[it].effort,
                ymin=exp_found,
                ymax=act_found,
                linestyles="dashed",
                color=color,
                label=f"{crit_name} WSS: {(wss*100):.1f} %, "
                f"Recall: {(recall*100):.1f} %",
            )

    def _graph_setup(self, simulation=True) -> None:
        true_pos = self.dataset_stats[self.it].pos_count
        dataset_size = self.dataset_stats[self.it].size

        plt.xlabel(f"number of read documents")
        plt.ylabel("number of retrieved relevant documents")
        if simulation:
            plt.title(
                f"Run on a dataset with {int(true_pos)} inclusions out of {int(dataset_size)}"
            )
        else:
            plt.title(f"Run on a dataset of {int(dataset_size)}")

    def _plot_static_data(self, recall_target: float) -> None:
        # Static data
        true_pos = self.dataset_stats[self.it].pos_count
        dataset_size = self.dataset_stats[self.it].size

        pos_target = int(np.ceil(recall_target * true_pos))
        effort_axis = self._effort_axis()

        plt.axhline(y=true_pos, linestyle=":", label=f"100 % recall ({true_pos})")
        plt.axhline(
            y=pos_target,
            linestyle=":",
            label=f"{int(recall_target * 100)} % recall ({pos_target})",
        )
        plt.plot(
            effort_axis,
            (effort_axis / dataset_size) * true_pos,
            ":",
            label=f"Exp. found at random",
        )

    def _plot_recall_stats(self, included: Optional[Sequence[str]] = list()) -> None:
        # Gather and reorganize recall data
        recall_stats = TemporalRecallStats.transpose_dict(self.recall_stats)
        # Plot pos docs docs found
        for name, stats in recall_stats.items():
            if included is None or name in included:
                self.plot_recall_statistic(
                    stats, "pos_docs_found", f"# found by {name}"
                )

    def _plot_estimators(
        self, included_estimators: Optional[Sequence[str]] = None
    ) -> None:
        if included_estimators is None:
            included_estimators = list(self.estimator_names)
        # Plotting estimations
        for i, estimator in enumerate(included_estimators):
            self._plot_estimator(estimator, color=f"C{i}")

    def _set_axes(
        self, x_lim: Optional[float] = None, y_lim: Optional[float] = None
    ) -> None:
        # Setting axis limitations
        true_pos = self.dataset_stats[self.it].pos_count
        if x_lim is not None:
            plt.xlim(0, x_lim)
        if y_lim is not None:
            plt.ylim(0, y_lim)
        else:
            plt.ylim(0, 1.4 * true_pos)

    def show(
        self,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        recall_target: float = 0.95,
        included_estimators: Optional[Sequence[str]] = None,
        included_models: Optional[Sequence[str]] = None,
        filename: "Optional[PathLike[str]]" = None,
    ) -> None:
        self._graph_setup()
        self._plot_static_data(recall_target)
        self._plot_recall_stats(included_models)
        self._plot_estimators(included_estimators)
        self._plot_stop_criteria()
        self._set_axes(x_lim, y_lim)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        plt.tight_layout()
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def user_show(
        self,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        included_estimators: Optional[Sequence[str]] = None,
        included_models: Optional[Sequence[str]] = None,
        filename: "Optional[PathLike[str]]" = None,
    ):
        self._graph_setup(simulation=False)
        self._plot_recall_stats(included_models)
        self._plot_estimators(included_estimators)
        self._plot_stop_criteria()
        self._set_axes(x_lim, y_lim)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        else:
            plt.show()
        plt.close()

    def wss_at_target(self, target: float) -> float:
        for it in self.it_axis:
            frame = self.recall_stats[it]
            if frame.recall >= target:
                return frame.wss
        return float("nan")

    def recall_at_stop(self, stop_criterion: str) -> float:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].recall
        return float("nan")

    def _it_at_stop(self, stop_criterion: str) -> Optional[int]:
        for it in self.it_axis:
            frame = self.stop_results[it]
            if frame[stop_criterion]:
                return it
        return None

    def wss_at_stop(self, stop_criterion: str) -> float:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].wss
        return float("nan")

    def relative_error(self, stop_criterion: str, recall_target: float) -> float:
        recall_at_stop = self.recall_at_stop(stop_criterion)
        relative_error = abs(recall_at_stop - recall_target) / recall_target
        return relative_error

    def loss_er_at_stop(self, stop_criterion: str) -> float:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].loss_er
        return float("nan")

    def effort_at_stop(self, stop_criterion: str) -> int:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].effort
        return self.recall_stats[self.it_axis[-1]].effort

    def proportional_effort_at_stop(self, stop_criterion: str) -> float:
        stop_it = self._it_at_stop(stop_criterion)
        if stop_it is not None:
            return self.recall_stats[stop_it].proportional_effort
        return 1.0


def filter_model_infos(
    mapping: Mapping[str, AbstractEstimator]
) -> Mapping[str, EstimationModelStatistics]:
    results = {
        key: est.model_info[-1]  # type: ignore
        for key, est in mapping.items()  # type: ignore
        if hasattr(est, "model_info")
    }  # type: ignore
    return results


class ModelStatsTar(TarExperimentPlotter[LT]):
    model_stats: ty.OrderedDict[int, Mapping[str, EstimationModelStatistics]]

    def __init__(self, pos_label: LT, neg_label: LT, dataset_name: str = "") -> None:
        super().__init__(pos_label, neg_label, dataset_name)
        self.model_stats = OrderedDict()

    def update(
        self,
        exp_iterator: ExperimentIterator[Any, Any, Any, Any, Any, LT],
        stop_result: Mapping[str, bool],
    ) -> None:
        super().update(exp_iterator, stop_result)
        self.model_stats[self.it] = filter_model_infos(exp_iterator.estimators)

    def _plot_estimators(
        self, included_estimators: Optional[Sequence[str]] = None
    ) -> None:
        super()._plot_estimators(included_estimators)
        if included_estimators is None:
            included_estimators = list(self.estimator_names)
        assert included_estimators is not None
        for estimator in included_estimators:
            if estimator in self.model_stats[self.it]:
                deviances = [
                    self.model_stats[it][estimator].deviance for it in self.it_axis
                ]
                plt.plot(
                    self._effort_axis(),
                    deviances,
                    label=f"Deviance {estimator}",
                    linestyle="--",
                )
