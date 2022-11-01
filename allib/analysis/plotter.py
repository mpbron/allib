import functools
from abc import ABC, abstractmethod
from os import PathLike
from typing import (
    Any,
    Dict,
    Generic,
    Iterable,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
)

import matplotlib.pyplot as plt  # type: ignore
import numpy as np  # type: ignore
import pandas as pd
from instancelib.analysis.base import classifier_performance
from instancelib.instances.base import InstanceProvider
from instancelib.labels.base import LabelProvider
from instancelib.typehints.typevars import KT

from ..activelearning import ActiveLearner
from ..activelearning.ensembles import AbstractEnsemble
from ..activelearning.insclass import ILMLBased
from ..estimation.base import AbstractEstimator
from ..utils.func import flatten_dicts
from .analysis import process_performance
from .experiments import ExperimentIterator  # type: ignore

LT = TypeVar("LT")


def name_formatter(learner: ActiveLearner[Any, Any, Any, Any, Any, Any]) -> str:
    name, label = learner.name
    if label is not None:
        return f"{name}_{label}"
    return name


class AbstractPlotter(ABC, Generic[LT]):
    @abstractmethod
    def update(self, activelearner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:
        raise NotImplementedError

    @abstractmethod
    def show(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError


class ExperimentPlotter(ABC, Generic[LT]):
    @abstractmethod
    def update(
        self,
        exp_iterator: ExperimentIterator[Any, Any, Any, Any, Any, LT],
        *args: Any,
        **kwargs: Any,
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def show(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

    @abstractmethod
    def print_last_stats(self) -> None:
        raise NotImplementedError


class ClassificationPlotter(AbstractPlotter[LT], Generic[KT, LT]):
    def __init__(
        self,
        test_set: InstanceProvider[Any, KT, Any, Any, Any],
        ground_truth: LabelProvider[KT, LT],
    ):
        self.test_set = list(test_set.values())
        self.ground_truth = ground_truth
        self.result_frame = pd.DataFrame()
        self.labelset = ground_truth.labelset

    def update(self, activelearner: ActiveLearner[Any, KT, Any, Any, Any, LT]) -> None:
        assert isinstance(activelearner, ILMLBased)
        results = classifier_performance(
            activelearner.classifier, self.test_set, self.ground_truth
        )
        stats: Dict[str, float] = dict()
        labeled = activelearner.env.labeled
        for label in self.labelset:
            seen_label_docs = activelearner.env.labels.get_instances_by_label(
                label
            ).intersection(labeled)
            gen_label_docs = activelearner.env.labels.get_instances_by_label(
                label
            ).difference(labeled)
            label_performance = results[label]
            stats[f"{label}_recall"] = label_performance.recall
            stats[f"{label}_precision"] = label_performance.precision
            stats[f"{label}_f1"] = label_performance.f1
            stats[f"{label}_docs"] = len(seen_label_docs)
            stats[f"{label}_ratio"] = len(seen_label_docs) / len(labeled)
            stats[f"{label}_n_generated"] = len(gen_label_docs)
        stats["n_generated_docs"] = len(activelearner.env.all_instances) - len(
            activelearner.env.dataset
        )
        stats["n_labeled"] = len(activelearner.env.labeled)
        result_it_frame = pd.DataFrame(stats)
        self.result_frame = (
            result_it_frame
            if not self.result_frame
            else pd.concat([self.result_frame, result_it_frame])
        )

    def show(
        self, metrics: Iterable[str] = ["f1", "recall"], filename: Optional[str] = None
    ) -> None:
        # Gathering intermediate results
        df = self.result_frame
        n_labeled = df["n_labeled"]
        for metric in metrics:
            for label in self.labelset:
                metric_values = df[f"{label}_{metric}"]
                plt.plot(n_labeled, metric_values, label=f"{label} :: {metric}")
        # Plotting positive document counts
        plt.xlabel(f"number of labeled instances")
        plt.ylabel(f"metric score")
        plt.title(f"Learning curves")
        plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")


class MultilabelPlotter(AbstractPlotter[LT], Generic[LT]):
    def __init__(self):
        self.result_frame = pd.DataFrame()

    def update(self, activelearner: ActiveLearner[Any, Any, Any, Any, Any, LT]) -> None:

        stats: Dict[str, float] = dict()
        stats["dataset_size"] = len(activelearner.env.dataset)
        for label in activelearner.env.labels.labelset:
            labeled_docs = activelearner.env.labels.get_instances_by_label(
                label
            ).intersection(activelearner.env.labeled)
            stats[f"{str(label)}_count"] = len(labeled_docs)
            stats[f"{str(label)}_true_size"] = float(
                activelearner.env.truth.document_count(label)
            )
        name = name_formatter(activelearner)
        results = {**stats}
        self.result_frame = self.result_frame.append(results, ignore_index=True)

    def show(
        self,
        x_lim: Optional[float] = None,
        y_lim: Optional[float] = None,
        all_estimations: bool = False,
        filename: Optional[PathLike] = None,
    ) -> None:
        pass
