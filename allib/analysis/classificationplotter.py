from dataclasses import dataclass
import itertools
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
    MutableMapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from ..estimation.base import AbstractEstimator, Estimate
from ..typehints import LT
from .experiments import ClassificationExperiment
from .plotter import ExperimentPlotter
from .statistics import DatasetStats
from instancelib.analysis.base import MulticlassModelMetrics

from dataclasses import dataclass
from .statistics import LabelALStatistics, ALStats

_T = TypeVar("_T")
_V = TypeVar("_V")
_W = TypeVar("_W")


def smooth_similar(
    xs: Union[npt.NDArray[Any], Sequence[_T]], ys: Union[npt.NDArray[Any], Sequence[_V]]
) -> Tuple[Sequence[_T], Sequence[_V]]:
    assert len(xs) == len(ys), f"len(xs={len(xs)}) != len(ys={len(ys)})"
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
    ), f"len(xs={len(xs)}) != len(ys={len(ys)}) != len(zs={len(zs)})"
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


class ClassificationPlotter(ExperimentPlotter[LT], Generic[LT]):
    dataset_name: str
    main_stats: ty.OrderedDict[int, Mapping[str, ALStats]]
    label_stats: ty.OrderedDict[int, Mapping[str, Mapping[LT, LabelALStatistics]]]
    metric_stats: ty.OrderedDict[int, Mapping[str, MulticlassModelMetrics]]
    stop_results: ty.OrderedDict[int, Mapping[str, Mapping[str, bool]]]

    def __init__(self, dataset_name: str = "") -> None:
        self.dataset_name = dataset_name

        self.dataset_stats = OrderedDict()
        self.metric_stats = OrderedDict()
        self.main_stats = OrderedDict()
        self.stop_results = OrderedDict()
        self.label_stats = OrderedDict()
        self.it = 0
        self.it_axis: List[int] = list()

    def update(
        self,
        exp_iterator: ClassificationExperiment[Any, Any, Any, Any, Any, LT],
        stop_result: Mapping[str, Mapping[str, bool]],
    ) -> None:
        self.it = exp_iterator.it
        self.it_axis.append(self.it)
        self.metric_stats[self.it] = exp_iterator.metrics
        self.label_stats[self.it] = exp_iterator.label_stats
        self.main_stats[self.it] = exp_iterator.main_stats
        self.stop_results[self.it] = exp_iterator.stop_result

    def print_last_stats(self) -> None:
        return

    def metric_names(self, model_name: str) -> FrozenSet[str]:
        return frozenset(["f1", "recall", "precision"])

    @property
    def model_names(self) -> FrozenSet[str]:
        if self.metric_stats:
            return frozenset(self.main_stats[self.it].keys())
        return frozenset()

    @property
    def criterion_names(self) -> FrozenSet[Tuple[str, str]]:
        if self.stop_results:
            return frozenset(
                [
                    (l_name, l_stop)
                    for l_name, l_stops in self.stop_results[self.it].items()
                    for l_stop in l_stops.keys()
                ]
            )
        return frozenset()

    def _plot_metrics(
        self,
        included_models: Optional[Sequence[str]] = None,
        included_metrics: Optional[Sequence[str]] = None,
        included_labels: Optional[Sequence[LT]] = None,
    ) -> None:
        model_selection = (
            included_models if included_models is not None else list(self.model_names)
        )
        i = 0
        for model in model_selection:
            metric_selection = (
                included_metrics
                if included_metrics is not None
                else list(self.metric_names(model))
            )
            label_selection = (
                included_labels
                if included_labels is not None
                else list(self.label_stats[self.it][model].keys())
            )
            effort_axis = self._effort_axis(model)

            combis = itertools.product(metric_selection, label_selection)
            for (metric, label) in combis:
                metrics = {
                    "f1": np.array(
                        [self.metric_stats[it][model][label].f1 for it in self.it_axis]
                    )
                    * 100,
                    "recall": np.array(
                        [
                            self.metric_stats[it][model][label].recall
                            for it in self.it_axis
                        ]
                    )
                    * 100,
                    "precision": np.array(
                        [
                            self.metric_stats[it][model][label].precision
                            for it in self.it_axis
                        ]
                    )
                    * 100,
                    "accuracy": np.array(
                        [
                            self.metric_stats[it][model][label].accuracy
                            for it in self.it_axis
                        ]
                    )
                    * 100,
                }
                color = f"C{i}"
                plt.plot(
                    effort_axis,
                    metrics[metric],
                    linestyle="-.",
                    label=f"{model} -- {metric} -- {label}",
                    color=color,
                )
                i += 1

    def _effort_axis(self, model: str) -> npt.NDArray[Any]:
        effort_axis = np.array(
            [
                self.main_stats[it][model].labeled / self.main_stats[it][model].dataset
                for it in self.it_axis
            ]
        )
        return effort_axis * 100

    def _graph_setup(self, simulation=True) -> None:
        model_name = next(iter(self.model_names))
        dataset_size = self.main_stats[self.it][model_name].dataset

        plt.xlabel(f"% annotated")
        plt.ylabel("result of the metric")

        if self.dataset_name:
            plt.title(
                f"Run on the `{self.dataset_name}` dataset with size {dataset_size}."
            )
        else:
            plt.title(f"Run on a dataset with size {dataset_size}.")

    def show(
        self,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        included_models: Optional[Sequence[str]] = None,
        included_metrics: Optional[Sequence[str]] = None,
        included_labels: Optional[Sequence[LT]] = None,
        filename: "Optional[PathLike[str]]" = None,
    ):
        self._graph_setup(simulation=False)
        self._plot_metrics(included_models, included_metrics, included_labels)
        # self._plot_stop_criteria()
        # self._set_axes(x_lim, y_lim)
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
