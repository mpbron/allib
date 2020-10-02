from __future__ import annotations
from typing import (Callable, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, TypeVar, Union)

import numpy as np # type: ignore

from ..environment import AbstractEnvironment
from ..instances import Instance, InstanceProvider
from ..machinelearning import AbstractClassifier

from .base import ActiveLearner, LabelPrediction
from .mostcertain import LabelMaximizer
from .poolbased import PoolbasedAL
from .random import RandomSampling
from .uncertainty import EntropySampling

KT = TypeVar("KT")

class InterleaveAL(PoolbasedAL, Generic[KT]):
    _name = "InterleaveAL"
    def __init__(self, classifier: AbstractClassifier) -> None:
        super().__init__(classifier)
        self._learners: Dict[str, PoolbasedAL]= {
            "random": RandomSampling(self.classifier),
            "uncertainty": EntropySampling(self.classifier),
            "mostcertain": LabelMaximizer(self.classifier, "Relevant"),
        }
        self._weights = (0.2, 0.2, 0.6)
    
    def __call__(self, environment: AbstractEnvironment) -> InterleaveAL:
        super().__call__(environment)
        assert self.env is not None
        for key, learner in self._learners.items():
            self._learners[key] = learner(self.env)
        self.initialized = True
        return self

    def _choose_active_learner(self) -> str:
        """ Makes a weighted decision and returns the chosen ActiveLearner """
        learners = list(self._learners.keys())
        learner_key = np.random.choice(learners, size=1, p=self._weights) # type: ignore
        return learner_key[0]

    def calculate_ordering(self) -> List[KT]:
        raise NotImplementedError

    def __next__(self) -> Instance:
        key = self._choose_active_learner()
        return next(self._learners[key])

    def retrain(self) -> None:
        super().retrain()
        for learner in self._learners.values():
            learner.fitted = True
            learner.ordering = None
