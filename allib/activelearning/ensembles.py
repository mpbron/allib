from __future__ import annotations

import logging
import random
from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, Generic, Iterable, List, Optional,
                    Sequence, Set, Tuple, TypeVar, Union)

import numpy as np  # type: ignore

from ..environment.base import AbstractEnvironment, IT
from instancelib.instances import Instance, InstanceProvider
from ..machinelearning import AbstractClassifier
from ..utils import get_random_generator
from .base import ActiveLearner
from .ml_based import MLBased
from .poolbased import PoolBasedAL
from .random import RandomSampling

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")

LOGGER = logging.getLogger(__name__)

class AbstractEnsemble(ABC, Generic[IT, KT, DT, VT, RT, LT]):
    _name = "AbstractEnsemble"
    learners: List[ActiveLearner[IT, KT, DT, VT, RT, LT]]
    env: AbstractEnvironment[IT, KT, DT, VT, RT, LT]
    _has_ordering: bool
    
    @abstractmethod
    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        """Internal functions that selects the next active learner for the next query

        Returns:
            ActiveLearner[IT, KT, DT, VT, RT, LT]: One of the learners from the ensemble
        """        
        raise NotImplementedError

    @property
    def has_ordering(self) -> bool:
        return self._has_ordering

    def update_ordering(self):
        """Updates the ordering for all learners of the ensemble
        """             
        for learner in self.learners:
            learner.update_ordering()
        self._has_ordering = True
        return True

    def __next__(self) -> IT:
        learner = self._choose_learner()
        return next(learner)

    @ActiveLearner.label_log
    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.env.unlabeled.discard(instance)
        self.env.labeled.add(instance)

    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as unlabeled
        
        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        self.env.labeled.discard(instance)
        self.env.unlabeled.add(instance)



class ManualEnsemble(AbstractEnsemble[IT, KT, DT, VT, RT, LT], PoolBasedAL[IT, KT, DT, VT, RT, LT],  Generic[IT, KT, DT, VT, RT, LT]):
    _name = "Ensemble"

    def __init__(self,
                 learners: List[ActiveLearner[IT, KT, DT, VT, RT, LT]],
                 probabilities: List[float], 
                 rng: Any = None, *_, identifier: Optional[str] = None, **__) -> None:
        super().__init__(identifier=identifier)
        self.learners: List[ActiveLearner[IT, KT, DT, VT, RT, LT]] = learners
        self.probabilities = probabilities
        self._sample_dict: Dict[KT, int] = {}
        self._rng: Any = get_random_generator(rng)

    def __call__(self, environment: AbstractEnvironment[IT, KT, DT, VT, RT, LT]) -> ManualEnsemble[IT, KT, DT, VT, RT, LT]:
        super().__call__(environment)
        for i, learner in enumerate(self.learners):
            env_copy = environment.from_environment(environment)
            self.learners[i] = learner(env_copy)
        self.initialized = True
        return self
    
    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        """Internal functions that selects the next active learner for the next query

        Returns
        -------
        ActiveLearner[IT, KT, DT, VT, RT, LT]
            One of the learners from the ensemble
        """
        idxs = np.arange(len(self.learners))
        al_idx: int = self._rng.choice(idxs, size=1, p=self.probabilities)[0]
        learner = self.learners[al_idx]
        return learner

    def __next__(self) -> IT:
        # Select the learner
        learner = self._choose_learner()
        
        # Select the next instance from the learner
        ins = next(learner)
        
        # Check if the instance identifier has not been labeled already
        while ins.identifier in self.env.labeled:
            # This instance has already been labeled my another learner.
            # Skip it and mark as labeled
            learner.set_as_labeled(ins)
            LOGGER.info(
                "The document with key %s was already labeled. Skipping", ins.identifier)
            learner = self._choose_learner()
            ins = next(learner)

        # Set the instances as sampled by learner with key al_idx and return the instance
        self._sample_dict[ins.identifier] = self.learners.index(learner)
        return ins

    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.env.labeled.add(instance)
        self.env.unlabeled.discard(instance)
        if instance.identifier in self._sample_dict:
            learner = self.learners[self._sample_dict[instance.identifier]]
            learner.set_as_labeled(instance)

    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.env.unlabeled.add(instance)
        self.env.labeled.discard(instance)
        if instance.identifier in self._sample_dict:
            learner = self.learners[self._sample_dict[instance.identifier]]
            learner.set_as_unlabeled(instance)
            del self._sample_dict[instance.identifier]

class StrategyEnsemble(AbstractEnsemble[IT, KT, DT, VT, RT, LT], 
                       MLBased[IT, KT, DT, VT, RT, LT, LVT, PVT], 
                       Generic[IT, KT, DT, VT, RT, LT, LVT, PVT]):
    _name = "StrategyEnsemble"

    def __init__(self,
                 classifier: AbstractClassifier[KT, VT, LT, LVT, PVT],
                 learners: List[ActiveLearner[IT, KT, DT, VT, RT, LT]],
                 probabilities: List[float], rng: Any = None, *_, 
                 identifier: Optional[str] = None, **__) -> None:
        super().__init__(classifier, RandomSampling())
        self.learners: List[ActiveLearner[IT, KT, DT, VT, RT, LT]] = learners
        self.probabilities = probabilities
        self._rng: Any = get_random_generator(rng)
        self.sampled: Set[KT] = set()
        self._has_ordering: bool = False 
    
    def __call__(self, environment: AbstractEnvironment[IT, KT, DT, VT, RT, LT]) -> StrategyEnsemble:
        """Initialize the learner with an environment

        Args:
            environment (AbstractEnvironment[KT, DT, VT, RT, LT]): the chosen environment

        Returns:
            StrategyEnsemble: The initialized environment
        """        
        super().__call__(environment)
        for learner in self.learners:
            learner(self.env)
        self.initialized = True
        return self

    def update_ordering(self) -> bool:
        successful = super().update_ordering()
        if successful:
            self.sampled = set()
        return successful

    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, VT, RT, LT]:
        """Internal functions that selects the next active learner for the next query

        Returns:
            ActiveLearner[IT, KT, DT, VT, RT, LT]: One of the learners from the ensemble
        """        
        idxs = np.arange(len(self.learners))
        al_idx: int = self._rng.choice(idxs, size=1, p=self.probabilities)[0]
        learner = self.learners[al_idx]
        return learner
    
    def __next__(self) -> IT:
        result = super().__next__()
        while result.identifier in self.sampled:
            result = super().__next__()
        self.sampled.add(result.identifier)
        return result