import typing as ty
from abc import ABC, abstractmethod
from collections import OrderedDict
from os import PathLike
from typing import (
    Any,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ..analysis.statistics import EstimationModelStatistics
from ..estimation.base import AbstractEstimator, Estimate
from ..typehints import LT
from .experiments import ExperimentIterator
from .plotter import ExperimentPlotter
from .statistics import TarDatasetStats, TemporalRecallStats
import scienceplots
import pylatex

_T = TypeVar("_T")
_V = TypeVar("_V")
_W = TypeVar("_W")


def escape(s: str, latex: bool = True) -> str:
    return pylatex.escape_latex(s) if latex else s


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

    def _plot_estimator(self, key: str, color="gray", alpha=0.2, latex=False) -> None:
        effort_axis = self._effort_axis()
        points = np.array([self.estimates[it][key].point for it in self.it_axis])
        lows = np.array([self.estimates[it][key].lower_bound for it in self.it_axis])
        uppers = np.array([self.estimates[it][key].upper_bound for it in self.it_axis])
        xs, ys = effort_axis, points #smooth_similar(effort_axis, points)
        xrs, ls, us = effort_axis, lows, uppers #smooth_similar3(effort_axis, lows, uppers)
        plt.plot(
            xs,
            ys,
            linestyle="-.",
            label=escape(f"Estimate by {key}", latex),
            color=color,
        )
        plt.fill_between(xrs, ls, us, color=color, alpha=alpha)  # type: ignore

    def _plot_stop_criteria(
        self, included_criteria: Optional[Sequence[str]], show_stats=True, latex=False
    ) -> None:
        results: Sequence[Tuple[int, float, float, str, str]] = list()
        if included_criteria is None:
            included_criteria = list(self.criterion_names)
        for i, crit_name in enumerate(included_criteria):
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
            nicer_name = crit_name.replace("_", " ").title()
            if show_stats:
                legend = (
                    f"{nicer_name} WSS: {(wss*100):.1f} Recall: {(recall*100):.1f} %"
                )
            else:
                legend = f"{nicer_name}"
            plt.vlines(
                x=self.recall_stats[it].effort,
                ymin=exp_found,
                ymax=act_found,
                linestyles="dashed",
                color=color,
                label=escape(legend, latex),
            )

    def _graph_setup(self, simulation=True, latex=False) -> None:
        if latex:
            plt.style.use(["science", "nature"])
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

    def _plot_static_data(self, recall_target: float, latex=False) -> None:
        # Static data
        true_pos = self.dataset_stats[self.it].pos_count
        dataset_size = self.dataset_stats[self.it].size

        pos_target = int(np.ceil(recall_target * true_pos))
        effort_axis = self._effort_axis()

        plt.axhline(
            y=true_pos, linestyle=":", label=escape(f"100 % recall ({true_pos})", latex)
        )
        plt.axhline(
            y=pos_target,
            linestyle=":",
            label=escape(f"{int(recall_target * 100)} % recall ({pos_target})", latex),
        )
        plt.plot(
            effort_axis,
            (effort_axis / dataset_size) * true_pos,
            ":",
            label=f"Exp. found at random",
        )

    def _plot_recall_stats(
        self, included: Optional[Sequence[str]] = list(), short_names=False, latex=False
    ) -> None:
        # Gather and reorganize recall data
        recall_stats = TemporalRecallStats.transpose_dict(self.recall_stats)
        # Plot pos docs docs found
        for name, stats in recall_stats.items():
            if short_names:
                try:
                    pname = name.split("-")[0].rstrip().lstrip()
                except:
                    pname = name
            else:
                pname = name
            if included is None or name in included:
                self.plot_recall_statistic(
                    stats, "pos_docs_found", escape(f"# by {pname}", latex)
                )

    def _plot_estimators(
        self, included_estimators: Optional[Sequence[str]] = None, latex=False
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

    def _plot_legend(self, latex=False) -> None:
        if latex:
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
        else:
            plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left", fontsize="xx-small")

    def show(
        self,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        recall_target: float = 0.95,
        included_estimators: Optional[Sequence[str]] = None,
        included_models: Optional[Sequence[str]] = None,
        included_stopcriteria: Optional[Sequence[str]] = None,
        filename: "Optional[PathLike[str]]" = None,
        latex: bool = False,
        show_stats=True,
        short_names=False,
    ) -> None:
        self._graph_setup(latex=latex)
        self._plot_static_data(recall_target, latex=latex)
        self._plot_recall_stats(included_models, latex=latex, short_names=short_names)
        self._plot_estimators(included_estimators, latex=latex)
        self._plot_stop_criteria(
            included_stopcriteria, latex=latex, show_stats=show_stats
        )
        self._set_axes(x_lim, y_lim)
        self._plot_legend(latex=latex)
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
        included_stopcriteria: Optional[Sequence[str]] = None,
        filename: "Optional[PathLike[str]]" = None,
        latex: bool = False,
        short_names=False,
    ):
        self._graph_setup(simulation=False, latex=latex)
        self._plot_recall_stats(included_models, latex=latex, short_names=short_names)
        self._plot_estimators(included_estimators, latex=latex)
        self._plot_stop_criteria(included_stopcriteria, latex=latex, show_stats=False)
        self._set_axes(x_lim, y_lim)
        self._plot_legend(latex=latex)
        plt.tight_layout()
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
        self, included_estimators: Optional[Sequence[str]] = None, latex=False
    ) -> None:
        super()._plot_estimators(included_estimators, latex)
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
                    label=escape(f"Deviance {estimator}", latex),
                    linestyle="--",
                )
