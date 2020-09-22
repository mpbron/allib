from __future__ import annotations
from typing import (Callable, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, TypeVar, Union)
import random
import numpy as np

from environment import AbstractEnvironment
from instances import Instance, InstanceProvider
from machinelearning import AbstractClassifier

from .base import ActiveLearner, Prediction
#from .elastic_query import ElasticQuery
from .mostcertain import MostCertainSampling
from .poolbased import PoolbasedAL
from .random import RandomSampling
from .uncertainty import LeastConfidence


class Estimator(PoolbasedAL):
    def __init__(self, learners: List[ActiveLearner], classifier) -> None:
        super().__init__(classifier)
        self._environment = None
        self._learners = learners
        self._dataset = None
        
    def __call__(self, environment: AbstractEnvironment) -> Estimator:
        super().__call__(environment)
        self._environment = environment
        for key, learner in self._learners.items():
            self._learners[key] = learner(self._environment)
        self.initialized = True
        self._dataset = environment.dataset_provider
        return self

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
        active_learner = random.choice(self._learners)
        return active_learner.query()

    def query_batch(self, batch_size: int) -> List[Instance]:
        active_learner = random.choice(self._learners)
        return active_learner.query_batch(batch_size)

    def relevance_predictions(self):
        feature_matrix = self._dataset.feature_matrix.matrix
        preds = np.hstack([learner._classifier.predict(feature_matrix) for learner in self._learners])
        return preds
        
