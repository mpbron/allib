from __future__ import annotations
from typing import (Callable, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, TypeVar, Union)

import numpy as np

from environment import AbstractEnvironment
from instances import Instance, InstanceProvider
from machinelearning import AbstractClassifier

from .base import ActiveLearner, Prediction
#from .elastic_query import ElasticQuery
from .mostcertain import LabelMaximizer
from .poolbased import PoolbasedAL
from .random import RandomSampling
from .uncertainty import EntropySampling

KT = TypeVar("KT")

class InterleaveAL(PoolbasedAL, Generic[KT]):
    _name = "InterleaveAL"
    def __init__(self, classifier: AbstractClassifier) -> None:
        super().__init__(classifier)
        self._environment = None
        self._learners = {
            "random": RandomSampling(self.classifier),
            "uncertainty": EntropySampling(self.classifier),
            "mostcertain": LabelMaximizer(self.classifier, "Relevant"),
            # "elastic_query": ElasticQuery(self.classifier)
        }
        self._weights = (0.2, 0.2, 0.6)
    
    def __call__(self, environment: AbstractEnvironment) -> InterleaveAL:
        super().__call__(environment)
        self._environment = environment
        for key, learner in self._learners.items():
            self._learners[key] = learner(self._environment)
        self.initialized = True
        return self

    def _choose_active_learner(self) -> str:
        """ Makes a weighted decision and returns the chosen ActiveLearner """
        learners = list(self._learners.keys())
        learner_key = np.random.choice(learners, 1, p=self._weights)
        return learner_key[0]

    def calculate_ordering(self) -> List[KT]:
        raise NotImplementedError

    def __next__(self) -> Instance:
        key = self._choose_active_learner()
        return next(self._learners[key])

    def query(self) -> Optional[Instance]:
        """Select the document whose  posterior probability is most near 0.5

        Returns
        -------
        Optional[Instance]
            An object containing:
                - The document identifier
                - The document vector
                - The document data
        """
        key = self._choose_active_learner()
        return self._learners[key].query()

    def query_batch(self, batch_size: int) -> List[Instance]:
        results = []
        for _ in range(batch_size):
            key = self._choose_active_learner()
            result = next(self._learners[key])
            results.append(result)
        return results

    def retrain(self) -> None:
        super().retrain()
        for learner in self._learners.values():
            learner.fitted = True
            learner.ordering = None
