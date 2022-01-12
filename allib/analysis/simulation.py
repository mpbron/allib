from dataclasses import dataclass
import functools
import itertools
import random
from __future__ import annotations
from abc import ABC, abstractmethod
from os import PathLike
from typing import (Any, Dict, FrozenSet, Generic, List, Mapping, Optional, Sequence,
                    Tuple, TypeVar, Union)

import instancelib as il
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd

from instancelib.feature_extraction.base import BaseVectorizer
from instancelib.functions.vectorize import vectorize
from instancelib.instances.base import Instance

from ..activelearning.base import ActiveLearner
from ..activelearning.ensembles import AbstractEnsemble
from ..analysis.analysis import process_performance
from ..environment.base import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from ..estimation.base import AbstractEstimator, Estimate
from ..factory.factory import ObjectFactory
from ..module.component import Component
from ..stopcriterion.base import AbstractStopCriterion
from ..typehints import DT, IT, KT, LT, RT, VT
from ..utils.func import flatten_dicts, prepend_keys  # type: ignore
from .initialization import Initializer
from .plotter import AbstractPlotter, name_formatter


def reset_environment(vectorizer: BaseVectorizer[IT], 
                      environment: AbstractEnvironment[IT, KT, DT, np.ndarray, RT, LT]
                      ) -> AbstractEnvironment[IT, KT, DT, np.ndarray, RT, LT]:
    env = MemoryEnvironment.from_environment_only_data(environment)
    vectorize(vectorizer, env, True, 200)
    return env

def initialize(factory: ObjectFactory,
               al_config: Dict[str, Any], 
               fe_config: Dict[str, Any],
               initializer: Initializer[IT, KT, LT], 
               env: AbstractEnvironment[IT, KT, DT, np.ndarray, DT, LT]
              ) -> Tuple[ActiveLearner[IT, KT, DT, np.ndarray, DT, LT],
                            BaseVectorizer[Instance[KT, DT, np.ndarray, DT]]]:
    """Build and initialize an Active Learning method.

    Parameters
    ----------
    factory : ObjectFactory
        The factory method that builds the components
    al_config : Dict[str, Any]
        The dictionary that declares the configuration of the Active Learning component
    fe_config : Dict[str, Any]
        The dictionary that declares the configuration of the Feature Extraction component
    initializer : Initializer[KT, LT]
        The function that determines how and which initial knowledge should be supplied to
        the Active Learner
    env : AbstractEnvironment[KT, DT, np.ndarray, DT, LT]
        The environment on which we should simulate

    Returns
    -------
    Tuple[ActiveLearner[KT, DT, np.ndarray, DT, LT], BaseVectorizer[Instance[KT, DT, np.ndarray, DT]]]
        A tuple that contains:

        - An :class:`~allib.activelearning.base.ActiveLearner` object according 
            to the configuration in `al_config`
        - An :class:`~allib.feature_extraction.base.BaseVectorizer` object according 
            to the configuration in `fe_config`
    """    
    # Build the active learners and feature extraction models
    learner: ActiveLearner[IT, KT, DT, np.ndarray, DT, LT] = factory.create(
        Component.ACTIVELEARNER, **al_config)
    vectorizer: BaseVectorizer[IT] = factory.create(
        Component.FEATURE_EXTRACTION, **fe_config)
    
    ## Copy the data to memory
    start_env = reset_environment(vectorizer, env)

    # Attach the environment to the active learner
    learner = learner(start_env)

    # Initialize the learner with initial knowledge
    learner = initializer(learner)
    return learner, vectorizer


class ExperimentIterator(Generic[IT, KT, DT, VT, RT, LT]):
    learner: ActiveLearner[IT, KT, DT, VT, RT, LT]
    it: int
    batch_size: int
    stop_interval: int
    stopping_criteria: Mapping[str,AbstractStopCriterion[LT]]
    estimators: Mapping[str, AbstractEstimator[IT, KT, DT, VT, RT, LT]]

    def __init__(self, 
                 learner: ActiveLearner[IT, KT, DT, VT, RT ,LT],
                 pos_label: LT,
                 neg_label: LT,
                 stopping_criteria: Mapping[str,AbstractStopCriterion[LT]],
                 estimators: Mapping[str, AbstractEstimator[IT, KT, DT, VT, RT, LT]],
                 batch_size: int = 1, 
                 stop_interval: Union[int, Mapping[str, int]] = 1,
                 estimation_interval: Union[int, Mapping[str, int]] = 1) -> None:
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

    def iter_increment(self) -> None:
        self.it += 1

    def __call__(self) -> Mapping[str, bool]:
        self._retrain()
        self._query_and_label()
        self._estimate_recall()
        stop_result = self.determine_stop()
        self.iter_increment()
        return stop_result

        
class ExperimentPlotter(ABC, Generic[LT]):
    data : Dict[int, Dict[str, Any]]

    @abstractmethod
    def update(self,
               exp_iterator: ExperimentIterator[Any, Any, Any, Any, Any, LT]
               ) -> None:
        raise NotImplementedError
    
    @abstractmethod
    def show(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError

@dataclass
class DatasetStats:
    pos_count: int
    neg_count: int
    size: int
    prevalence: float

    @classmethod
    def from_learner(cls, 
                     learner: ActiveLearner[Any, Any, Any, Any, Any, LT],
                     pos_label: LT,
                     neg_label: LT) -> DatasetStats:
        pos_count = learner.env.truth.document_count(pos_label)
        neg_count = learner.env.truth.document_count(neg_label)
        size = len(learner.env.dataset)
        prevalence = pos_count / size
        return DatasetStats(pos_count, neg_count, size, prevalence)

@dataclass
class TemporalRecallStats:
    name: float
    wss: float
    recall: float
    pos_docs_found: int
    neg_docs_found: int
    effort: int
    child_statistics: Sequence[Tuple[str, TemporalRecallStats]]

    @classmethod
    def from_learner(cls,  
                     learner: ActiveLearner[Any, Any, Any, Any, Any, LT],
                     pos_label: LT,
                     neg_label: LT
                     ) -> TemporalRecallStats:
        perf = process_performance(learner, pos_label)
        pos_docs = learner.env.get_subset_by_labels(learner.env.labeled, pos_label)
        neg_docs = learner.env.get_subset_by_labels(learner.env.labeled, neg_label)
        effort = len(learner.env.labeled)
        if isinstance(learner, AbstractEnsemble):
            subresults = tuple([(learner.name, cls.from_learner(learner)) for learner  in learner.learners])
        else:
            subresults = tuple([])
        return TemporalRecallStats(perf.wss, perf.recall, pos_docs, neg_docs, effort, subresults)


class TarExperimentPlotter(ExperimentPlotter[LT], Generic[LT]):
    pos_label: LT
    neg_label: LT

    dataset_stats: Dict[int, DatasetStats]
    recall_stats: Dict[int, TemporalRecallStats]
    estimates: Dict[int, Dict[str, Estimate]]
    stop_results: Dict[int, Dict[str, bool]]

    def __init__(self, pos_label: LT, neg_label: LT) -> None:
        self.pos_label = pos_label
        self.neg_label = neg_label
        
        self.dataset_stats = dict()
        self.recall_stats = dict()
        self.estimates = dict()
        self.stop_results = dict()
        self.it = 0

    def update(self, 
               exp_iterator: ExperimentIterator[Any, Any, Any, Any, Any, LT],
               stop_result: Mapping[str, bool]) -> None:
        learner = exp_iterator.learner
        self.it = exp_iterator.it
        self.recall_stats[self.it]  = TemporalRecallStats.from_learner(learner, self.pos_label, self.neg_label)
        self.dataset_stats[self.it] = DatasetStats.from_learner(learner, self.pos_label, self.neg_label)
        self.estimates[self.it] = {name: estimate for name, estimate in exp_iterator.recall_estimate.items()}
        self.stop_results[self.it] = stop_result
        
       
    def show(self,
             x_lim: Optional[float] = None,
             y_lim: Optional[float] = None,
             all_estimations: bool = False,
             filename: "Optional[PathLike[str]]" = None) -> None:
        # Gathering intermediate results
        x_size = len(self.dataset_stats)
        df = pd.DataFrame.from_dict(self.data, orient="index")

        n_pos_overall = self.dataset_stats[self.it].pos_count
        n_found = self.recall_stats[self.it].pos_docs_found

        true_pos = self.dataset_stats[self.it].pos_count
        effort = self.recall_stats[self.it].effort
        total_n = self.dataset_stats[self.it].size
        
        wss = np.array(df.wss)[-1] * 100
        recall = np.array(df.recall)[-1] * 100
        n_exp = int(np.floor((effort / total_n) * true_pos))

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
                     label="# found by $\\mathcal{C}" f"_{i}$ ({total_learner})")
        
        estimator_names = self._get_estimator_cols(df)
        for estimator_name in estimator_names:
            points = np.array(df[f"Estimation_{estimator_name}_point"])
            lows = np.array(df[f"Estimation_{estimator_name}_low"])
            uppers = np.array(df[f"Estimation_{estimator_name}_up"])
            plt.plot(n_documents_read, points, "-.", label="Estimate") # type: ignore
            plt.fill_between(n_documents_read, # type: ignore
                             lows, uppers, color='gray', alpha=0.2)
        
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
        plt.xlabel(f"number of read documents (total {effort}), WSS = {wss:.2f} % Recall = {recall:.2f} %")
        plt.ylabel("number of documents")
        plt.title(f"Run on a dataset with {int(true_pos)} inclusions out of {int(total_n)}")
        if not all_estimations:
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

        if filename is not None:
            plt.savefig(filename, bbox_inches='tight')

    def wss_at_target(self, target: float) -> float:
        for t, frame in self.data.items():
            if frame["recall"] >= target:
                return frame["wss"]
        return float("nan")

    def recall_at_stop(self, stop_criterion: str) -> float:
        for t, frame in self.data.items()
        


class TarSimulator(Generic[IT, KT, DT, VT, RT, LT]):
    plotter: ExperimentPlotter[LT]
    experiment: ExperimentIterator

    def __init__(self, experiment: ExperimentPlotter, 
                       plotter: ExperimentPlotter,
                       max_it: Optional[int]=None) -> None:
        self.experiment = experiment
        self.plotter = plotter
        self.max_it = max_it
    
    @property
    def _debug_finished(self) -> bool:
        if self.max_it is None:
            return False
        return self.experiment.it > self.max_it
     
    def simulate(self) -> None:
        while not self.experiment.finished and not self._debug_finished:
            result = self.experiment()
            self.plotter.update(self.experiment)
    
   





def simulate(learner: ActiveLearner[IT, KT, DT, VT, RT, LT],
             stop_crit: AbstractStopCriterion[LT],
             plotter: AbstractPlotter[LT],
             batch_size: int) -> Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT],
                         AbstractPlotter[LT]]:
    """Simulates the Active Learning 

    Parameters
    ----------
    learner : ActiveLearner[IT, KT, DT, VT, RT, LT]
        [description]
    stop_crit : AbstractStopCriterion[LT]
        [description]
    plotter : BinaryPlotter[LT]
        [description]
    batch_size : int
        [description]

    Returns
    -------
    Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT], BinaryPlotter[LT]]
        [description]
    """    
    while not stop_crit.stop_criterion:
        # Train the model
        learner.update_ordering()
        # Sample batch_size documents from the learner
        sample = itertools.islice(learner, batch_size)
        for instance in sample:
            # Retrieve the labels from the oracle
            oracle_labels = learner.env.truth.get_labels(instance)

            # Set the labels in the active learner
            learner.env.labels.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)

        plotter.update(learner)
        stop_crit.update(learner)
    
    return learner, plotter

def simulate_stop_iteration(learner: ActiveLearner[IT, KT, DT, VT, RT, LT],
             stop_crit: AbstractStopCriterion[LT],
             plotter: AbstractPlotter[LT],
             batch_size: int,
             check_stop: int = 10) -> Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT],
                         AbstractPlotter[LT]]:
    """Simulates the Active Learning 

    Parameters
    ----------
    learner : ActiveLearner[IT, KT, DT, VT, RT, LT]
        [description]
    stop_crit : AbstractStopCriterion[LT]
        [description]
    plotter : BinaryPlotter[LT]
        [description]
    batch_size : int
        [description]

    Returns
    -------
    Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT], BinaryPlotter[LT]]
        [description]
    """
    it = 0
    while not stop_crit.stop_criterion:
        # Train the model
        learner.update_ordering()
        # Sample batch_size documents from the learner
        sample = itertools.islice(learner, batch_size)
        for instance in sample:
            # Retrieve the labels from the oracle
            oracle_labels = learner.env.truth.get_labels(instance)

            # Set the labels in the active learner
            learner.env.labels.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)
            it = it + 1

        if it % check_stop == 0:
            plotter.update(learner)
            stop_crit.update(learner)
    
    return learner, plotter

def multilabel_all_non_empty(learner: ActiveLearner[Any, Any, Any, Any, Any, Any], count: int) -> bool:
    provider = learner.env.labels
    non_empty = all(
        [provider.document_count(label) > count for label in provider.labelset])
    return non_empty

def simulate_with_cold_start(learner: ActiveLearner[IT, KT, DT, VT, RT, LT],
             stop_crit: AbstractStopCriterion[LT],
             plotter: AbstractPlotter[LT],
             batch_size: int, start_count=2) -> Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT],
                         AbstractPlotter[LT]]:
    """Simulates the Active Learning 

    Parameters
    ----------
    learner : ActiveLearner[IT, KT, DT, VT, RT, LT]
        [description]
    stop_crit : AbstractStopCriterion[LT]
        [description]
    plotter : BinaryPlotter[LT]
        [description]
    batch_size : int
        [description]

    Returns
    -------
    Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT], BinaryPlotter[LT]]
        [description]
    """
    learner.update_ordering()
    while not multilabel_all_non_empty(learner, start_count):
        instance = next(learner)
        oracle_labels = learner.env.truth.get_labels(instance)
        # Set the labels in the active learner
        learner.env.labels.set_labels(instance, *oracle_labels)
        learner.set_as_labeled(instance)
    while not stop_crit.stop_criterion:
        # Train the model
        learner.update_ordering()
        # Sample batch_size documents from the learner
        sample = itertools.islice(learner, batch_size)
        for instance in sample:
            # Retrieve the labels from the oracle
            oracle_labels = learner.env.truth.get_labels(instance)

            # Set the labels in the active learner
            learner.env.labels.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)

        plotter.update(learner)
        stop_crit.update(learner)
    
    return learner, plotter

def simulate_classification(learner: ActiveLearner[IT, KT, DT, VT, RT, LT],
             stop_crit: AbstractStopCriterion[LT],
             plotter: AbstractPlotter[LT],
             batch_size: int, start_count=2) -> Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT],
                         AbstractPlotter[LT]]:
    """Simulates the Active Learning procedure

    Parameters
    ----------
    learner : ActiveLearner[IT, KT, DT, VT, RT, LT]
        The Active Learning object
    stop_crit : AbstractStopCriterion[LT]
        The stopping criterion
    plotter : BinaryPlotter[LT]
        A plotter that tracks the results
    batch_size : int
        The batch size of each sample 
    start_count : int
        The number of instances that each class recieves before training the classification process. 

    Returns
    -------
    Tuple[ActiveLearner[IT, KT, DT, VT, RT, LT], AbstractPlotter[LT]]
        A tuple consisting of the final model and the plot of the process
    """
    learner.update_ordering()
    while not multilabel_all_non_empty(learner, start_count):
        instance = next(learner)
        oracle_labels = learner.env.truth.get_labels(instance)
        # Set the labels in the active learner
        learner.env.labels.set_labels(instance, *oracle_labels)
        learner.set_as_labeled(instance)
    while not stop_crit.stop_criterion:
        # Train the model
        learner.update_ordering()
        # Sample batch_size documents from the learner
        sample = itertools.islice(learner, batch_size)
        for instance in sample:
            # Retrieve the labels from the oracle
            oracle_labels = learner.env.truth.get_labels(instance)
            print(instance)
            print(oracle_labels)
            # Set the labels in the active learner
            learner.env.labels.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)

        plotter.update(learner)
        stop_crit.update(learner)
    
    return learner, plotter

