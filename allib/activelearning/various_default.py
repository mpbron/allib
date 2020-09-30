from __future__ import annotations
from typing import (Callable, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, TypeVar, Union)

from numpy.random import choice # type: ignore

from ..environment import AbstractEnvironment
from ..instances import Instance, InstanceProvider
from ..machinelearning import AbstractClassifier

from .base import ActiveLearner
from .mostcertain import MostConfidence
from .poolbased import PoolbasedAL
from .random import RandomSampling
from .uncertainty import EntropySampling


class VariousDjangoDefault(PoolbasedAL):
    _name = "VariousDjangoDefault"

    def __init__(self, classifier: AbstractClassifier, min_train_annotations=10) -> None:
        super().__init__(classifier)
        self._environment = None
        self._pos_examples_threshold = min_train_annotations
        self._learners: Dict[str, PoolbasedAL]= {
            "random": RandomSampling(self.classifier),
            "uncertainty": EntropySampling(self.classifier),
            "mostconfidence": MostConfidence(self.classifier),
            #"elastic_query": ElasticQuery(self.classifier)
        }
        self._weights_pos = (0.2, 0.1, 0.7)
        self._weights_neg = (1.0, 0.0, 0.0)

    def __call__(self, environment: AbstractEnvironment) -> VariousDjangoDefault:
        super().__call__(environment)
        for key, learner in self._learners.items():
            self._learners[key] = learner(self.env)
        self.initialized = True
        return self

    def calculate_ordering(self):
        # Should not be called
        raise NotImplementedError

    def _choose_active_learner(self) -> ActiveLearner:
        """ Makes a weighted decision and returns the chosen ActiveLearner """
        learners = list(self._learners.keys())
        if self._labelprovider.len_positive >= self._pos_examples_threshold:
            learner_key = choice(learners, 1, p=self._weights_pos)
        else:
            learner_key = choice(learners, 1, p=self._weights_neg)
        return self._learners[learner_key[0]]

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
        active_learner = self._choose_active_learner()
        return active_learner.query()

    def query_batch(self, batch_size: int) -> List[Instance]:
        active_learner = self._choose_active_learner()
        return active_learner.query_batch(batch_size)

    def __next__(self) -> Instance:
        learner = self._choose_active_learner()
        return next(learner)
