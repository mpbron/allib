from __future__ import annotations

import pickle
from pathlib import Path
from typing import (
    Any,
    Callable,
    Generic,
    Mapping,
    Optional,
    Tuple,
)

import numpy.typing as npt
from instancelib.feature_extraction.base import BaseVectorizer
from instancelib.functions.vectorize import vectorize
from instancelib.instances.base import Instance
from tqdm.auto import tqdm

from ..activelearning.base import ActiveLearner
from ..environment.base import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from ..factory.factory import ObjectFactory
from ..module.component import Component
from ..typehints import DT, IT, KT, LT, RT, VT
from .classificationplotter import ClassificationPlotter
from .experiments import ClassificationExperiment, ExperimentIterator
from .initialization import Initializer
from .tarplotter import TarExperimentPlotter


def initialize_tar_simulation(
    factory: ObjectFactory,
    al_config: Mapping[str, Any],
    fe_config: Mapping[str, Any],
    initializer: Initializer[IT, KT, LT],
    env: AbstractEnvironment[IT, KT, DT, npt.NDArray[Any], RT, LT],
    pos_label: LT,
    neg_label: LT,
) -> Tuple[
    ActiveLearner[IT, KT, DT, npt.NDArray[Any], RT, LT],
    Optional[BaseVectorizer[Instance[KT, DT, npt.NDArray[Any], RT]]],
]:
    """Build and initialize an Active Learning method.

    Parameters
    ----------
    factory : ObjectFactory
        The factory method that builds the components
    al_config : Mapping[str, Any]
        The dictionary that declares the configuration of the Active Learning component
    fe_config : Mapping[str, Any]
        The dictionary that declares the configuration of the Feature Extraction component
    initializer : Initializer[IT, KT, LT]
        The function that determines how and which initial knowledge should be supplied to
        the Active Learner
    env : AbstractEnvironment[KT, DT, npt.NDArray[Any], DT, LT]
        The environment on which we should simulate
    pos_label : LT
        The label of the positive class
    neg_label : LT

    Returns
    -------
    Tuple[ActiveLearner[KT, DT, npt.NDArray[Any], DT, LT], BaseVectorizer[Instance[KT, DT, npt.NDArray[Any], DT]]]
        A tuple that contains:

        - An :class:`~allib.activelearning.base.ActiveLearner` object according
            to the configuration in `al_config`
        - An :class:`~allib.feature_extraction.base.BaseVectorizer` object according
            to the configuration in `fe_config`
    """
    # Get the active learner builder and feature extraction models
    learner_builder: Callable[
        ..., ActiveLearner[IT, KT, DT, npt.NDArray[Any], RT, LT]
    ] = factory.create(Component.ACTIVELEARNER, **al_config)
    if fe_config:
        vectorizer: BaseVectorizer[
            Instance[KT, DT, npt.NDArray[Any], RT]
        ] = factory.create(Component.FEATURE_EXTRACTION, **fe_config)
        vectorize(vectorizer, env, True, 2000)
    else:
        vectorizer = None
    ## Copy the data to memory
    start_env = MemoryEnvironment.from_environment_only_data(env)

    # Build the Active Learner object
    learner = learner_builder(start_env, pos_label=pos_label, neg_label=neg_label)

    # Initialize the learner with initial knowledge
    learner = initializer(learner)
    return learner, vectorizer


class TarSimulator(Generic[IT, KT, DT, VT, RT, LT]):
    plotter: TarExperimentPlotter[LT]
    experiment: ExperimentIterator
    output_pkl_path: Optional[Path]
    output_pdf_path: Optional[Path]
    plot_interval: int

    def __init__(
        self,
        experiment: ExperimentIterator[IT, KT, DT, VT, RT, LT],
        plotter: TarExperimentPlotter[LT],
        max_it: Optional[int] = None,
        print_enabled=False,
        output_path: Optional[Path] = None,
        output_pdf_path: Optional[Path] = None,
        plot_interval: int = 20,
    ) -> None:
        self.experiment = experiment
        self.plotter = plotter
        self.max_it = max_it
        self.print_enabled = print_enabled
        self.output_pkl_path = output_path
        self.output_pdf_path = output_pdf_path
        self.plot_interval = plot_interval

    @property
    def _debug_finished(self) -> bool:
        if self.max_it is None:
            return False
        return self.experiment.it > self.max_it

    def simulate(self) -> None:
        with tqdm(total=len(self.experiment.learner.env.dataset)) as pbar:
            pbar.update(self.experiment.learner.len_labeled)
            while not self.experiment.finished and not self._debug_finished:
                result = self.experiment()
                self.plotter.update(self.experiment, result)
                pbar.update(1)
                found = self.plotter.recall_stats[self.plotter.it].pos_docs_found
                estimates = [
                    f"{n} {e.point:.1f}, CI: [{e.lower_bound:.1f}, {e.upper_bound:.1f}]"
                    for n, e in self.plotter.estimates[self.plotter.it].items()
                ]
                pbar.set_description(f"Found: {found}, Estimate: {estimates}")
                if self.output_pkl_path is not None:
                    with self.output_pkl_path.open("wb") as fh:
                        pickle.dump(self.plotter, fh)
                if (
                    self.experiment.it % self.plot_interval == 0
                    and self.output_pdf_path is not None
                ):
                    self.plotter.show(filename=self.output_pdf_path)


class ClassificationSimulator(Generic[IT, KT, DT, VT, RT, LT]):
    plotter: ClassificationPlotter[LT]
    experiment: ClassificationExperiment[IT, KT, DT, VT, RT, LT]

    def __init__(
        self,
        experiment: ClassificationExperiment[IT, KT, DT, VT, RT, LT],
        plotter: ClassificationPlotter[LT],
        max_it: Optional[int] = None,
        print_enabled=False,
    ) -> None:
        self.experiment = experiment
        self.plotter = plotter
        self.max_it = max_it
        self.print_enabled = print_enabled

    @property
    def _debug_finished(self) -> bool:
        if self.max_it is None:
            return False
        return self.experiment.it > self.max_it

    def simulate(self) -> None:
        first_learner = next(iter(self.experiment.learners.values()))
        with tqdm(total=len(first_learner.env.dataset)) as pbar:
            pbar.update(first_learner.len_labeled)
            while not self.experiment.finished and not self._debug_finished:
                result = self.experiment()
                self.plotter.update(self.experiment, result)
                if self.print_enabled:
                    self.plotter.print_last_stats()
                pbar.update(1)


def multilabel_all_non_empty(
    learner: ActiveLearner[Any, Any, Any, Any, Any, Any], count: int
) -> bool:
    provider = learner.env.labels
    non_empty = all(
        [provider.document_count(label) > count for label in provider.labelset]
    )
    return non_empty
