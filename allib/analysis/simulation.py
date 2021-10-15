import itertools
import random
from typing import Any, Dict, List, Sequence, Tuple, TypeVar

import numpy as np  # type: ignore
from instancelib.feature_extraction.base import BaseVectorizer
from instancelib.functions.vectorize import vectorize
from instancelib.instances.base import Instance

from ..activelearning.base import ActiveLearner
from ..environment.base import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from ..factory.factory import ObjectFactory
from ..module.component import Component
from ..stopcriterion.base import AbstractStopCriterion
from ..typehints import DT, IT, KT, LT, RT, VT
from .initialization import Initializer
from .plotter import AbstractPlotter


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

