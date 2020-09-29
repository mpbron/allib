from __future__ import annotations
from typing import (Callable, Dict, Generic, Iterable, List, Optional,
                    Sequence, Tuple, TypeVar, Union)
import random
import numpy as np

from ..environment import AbstractEnvironment
from ..instances import Instance, InstanceProvider
from ..machinelearning import AbstractClassifier

from .base import ActiveLearner, Prediction, NotInitializedException
#from .elastic_query import ElasticQuery
from .mostcertain import MostCertainSampling
from .poolbased import PoolbasedAL
from .random import RandomSampling
from .uncertainty import LeastConfidence
from ..utils.random import get_random_generator


class Estimator(PoolbasedAL):
    def __init__(self, classifier, learners: List[PoolbasedAL], probabilities : List[int] = None, rng = None) -> None:
        super().__init__(classifier)
        self._environment = None
        self._learners: Dict[int, PoolbasedAL] = { i: learners for i, learners in enumerate(learners)}
        self._probabilities = [1.0 / len(learners)] * len(learners) if probabilities is None else probabilities
        self._dataset = None
        self._sample_dict: Dict[]= {}
        self._labeled_dict: Dict[int, PoolbasedAL]= {}
        self._rng = get_random_generator(rng)
        
    def __call__(self, environment: AbstractEnvironment) -> Estimator:
        super().__call__(environment)
        for key, learner in self._learners:
            env_copy = environment.from_environment(environment)
            self._learners[key] = learner(env_copy)
        self.initialized = True
        self._dataset = environment.dataset_provider
        return self
    
    def calculate_ordering(self):
        raise NotImplementedError

    def __next__(self) -> Instance:
        indices = np.range(len(self._learners))
        al_idx = self._rng.choice(indices, size = 1, p = self._probabilities)[0]
        learner = self._learners[al_idx]
        ins = next(learner)
        while ins.identifier in self._labeled:
            learner.set_as_labeled(ins)
            ins = next(learner)
        self._sample_dict[ins.identifier] = al_idx
        return ins

    def relevance_predictions(self):
        feature_matrix = self._dataset.feature_matrix.matrix
        preds = np.hstack([learner._classifier.predict(feature_matrix) for learner in self._learners])
        return preds
    
    def set_as_labeled(self, instance: Instance) -> None:
        super().set_as_labeled(instance)
        learner = self._learners[self._sample_dict[instance.identifier]]
        learner.set_as_labeled(instance)


    def retrain(self) -> None:
        # Ensure instances match labelings
        if not self.initialized:
            raise NotInitializedException
        for key, learner in self._learners:
            learner.retrain()
        instances = [instance for _, instance in self._labeled.items()]
        labelings = [self._labelprovider.get_labels(instance) for instance in instances]
        self.classifier.fit_instances(instances, labelings)
        self.fitted = True
        self.ordering = None