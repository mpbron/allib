from __future__ import annotations
from typing import (Dict, Generic, List, Optional,
                    TypeVar, Any, Tuple)

import logging
import os
import numpy as np  # type: ignore
import pandas as pd  # type: ignore

import rpy2.robjects as ro # type: ignore
from rpy2.robjects.packages import importr #type: ignore
from rpy2.robjects import pandas2ri # type: ignore
from rpy2.robjects.conversion import localconverter # type: ignore

from ..environment import AbstractEnvironment
from ..instances import Instance
from ..machinelearning import AbstractClassifier

from .base import NotInitializedException
from .poolbased import PoolbasedAL

from ..utils import get_random_generator

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")

LOGGER = logging.getLogger(__name__)


class Ensemble(PoolbasedAL[KT, DT, VT, RT, LT, LVT, PVT], Generic[KT, DT, VT, RT, LT, LVT, PVT]):
    _name = "Ensemble"

    def __init__(self,
                 classifier: AbstractClassifier[KT, VT, LT, LVT, PVT],
                 learners: List[PoolbasedAL[KT, DT, VT, RT, LT, LVT, PVT]],
                 pos_label: str = "Relevant",
                 probabilities: Optional[List[float]] = None, rng: Any = None, *_, **__) -> None:
        super().__init__(classifier)
        self._environment = None
        self._learners: Dict[int, PoolbasedAL[KT, DT, VT, RT, LT, LVT, PVT]] = {
            i: learners for i, learners in enumerate(learners)}
        self._probabilities = [
            1.0 / len(learners)] * len(learners) if probabilities is None else probabilities
        self._sample_dict: Dict[KT, int] = {}
        self._rng: Any = get_random_generator(rng)
        self.pos_label = pos_label

    def __call__(self,
                 environment: AbstractEnvironment[KT, DT, VT, RT, LT]
                 ) -> Ensemble[KT, DT, VT, RT, LT, LVT, PVT]:
        super().__call__(environment)
        for key, learner in self._learners.items():
            env_copy = environment.from_environment(environment)
            self._learners[key] = learner(env_copy)
        self.initialized = True
        return self

    def calculate_ordering(self) -> List[KT]:
        raise NotImplementedError

    def __next__(self) -> Instance[KT, DT, VT, RT]:
        # Choose the next random learners
        indices = np.arange(len(self._learners))
        al_idx = self._rng.choice(indices, size=1, p=self._probabilities)[0]
        learner = self._learners[al_idx]
        
        # Select the next instance from the learner
        ins = next(learner)
        
        # Check if the instance identifier has not been labeled already
        while ins.identifier in self.env.labeled:
            # This instance has already been labeled my another learner.
            # Skip it and mark as labeled
            learner.set_as_labeled(ins)
            LOGGER.info(
                "The document with key %s was already labeled. Skipping", ins.identifier)
            ins = next(learner)

        # Set the instances as sampled by learner with key al_idx and return the instance
        self._sample_dict[ins.identifier] = al_idx
        return ins

    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.env.labeled.add(instance)
        self.env.unlabeled.discard(instance)
        if instance.identifier in self._sample_dict:
            learner = self._learners[self._sample_dict[instance.identifier]]
            if self.pos_label in self.env.labels.get_labels(instance):
                for learner in self._learners.values():
                    learner.set_as_labeled(instance)
            else:
                learner.set_as_labeled(instance)
        else:
            for learner in self._learners.values():
                learner.set_as_labeled(instance)
        
    def retrain(self) -> None:
        if not self.initialized or self.env is None:
            raise NotInitializedException
        for learner in self._learners.values():
            learner.retrain()
        instances = [instance for _, instance in self.env.labeled.items()]
        labelings = [self.env.labels.get_labels(
            instance) for instance in instances]
        self.classifier.fit_instances(instances, labelings)
        self.fitted = True
        self.ordering = None

    def get_label_matrix(self, label: LT) -> pd.DataFrame:
        rows = {ins_key: {
            l_key: ins_key in learner.env.labeled
            for l_key, learner in self._learners.items()}
            for ins_key in self.env.labels.get_instances_by_label(label)
        }
        dataframe = pd.DataFrame.from_dict( # type: ignore
            rows, orient="index")  
        return dataframe
        

    