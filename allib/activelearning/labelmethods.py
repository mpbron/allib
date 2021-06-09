from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional,
                    Sequence, Tuple, TypeVar)

import numpy as np  # type: ignore

from ..environment import AbstractEnvironment
from ..machinelearning import AbstractClassifier
from ..typehints import DT, IT, KT, LT, RT
from .base import ActiveLearner
from .ensembles import AbstractEnsemble
from .ml_based import (AbstractSelectionCriterion, FeatureMatrix, MLBased,
                       ProbabilityBased)
from .random import RandomSampling


class LabelProbabilityBased(ProbabilityBased[IT, KT, DT, RT, LT], ABC, Generic[IT, KT, DT, RT, LT]):
    def __init__(self, 
                 classifier: AbstractClassifier[KT, np.ndarray, LT, np.ndarray, np.ndarray], 
                 selection_criterion: AbstractSelectionCriterion,
                 label: LT, fallback = RandomSampling[IT, KT, DT, np.ndarray, RT, LT](),
                 identifier: Optional[str] = None,  *_, **__) -> None:
        super().__init__(classifier, selection_criterion, fallback, identifier=identifier)
        self.label = label
        self.labelposition: Optional[int] = None

    def __call__(self, environment: AbstractEnvironment[IT, KT, DT, np.ndarray, RT, LT]) -> LabelProbabilityBased[IT, KT, DT, RT, LT]:
        super().__call__(environment)
        self.labelposition = self.classifier.get_label_column_index(self.label)
        return self

    @property
    def name(self) -> Tuple[str, LT]:
        if self.identifier is not None:
            return f"{self.identifier}", self.label
        return f"{self._name} :: {self.classifier.name}", self.label

    def _get_predictions(self, matrix: FeatureMatrix[KT]) -> Tuple[Sequence[KT], np.ndarray]:
        prob_vec: np.ndarray = self.classifier.predict_proba(
            matrix.matrix)  # type: ignore
        # type: ignore
        sliced_prob_vec: np.ndarray = prob_vec[:, self.labelposition] # type: ignore
        keys = matrix.indices
        return keys, sliced_prob_vec


class LabelEnsemble(AbstractEnsemble[IT, KT, DT, np.ndarray, RT, LT], MLBased[IT, KT,DT, np.ndarray, RT, LT, np.ndarray, np.ndarray],Generic[IT, KT, DT, RT, LT]):
    _name = "LabelEnsemble"

    def __init__(self,
                 classifier: AbstractClassifier[KT, np.ndarray, LT, np.ndarray, np.ndarray],
                 al_method: Callable[..., LabelProbabilityBased[IT, KT, DT, RT, LT]],
                 *_, **__
                 ) -> None:
        self._al_builder = al_method
        self.label_dict: Dict[LT, int] = dict()
        self.learners: List[ActiveLearner[IT, KT, DT, np.ndarray, RT, LT]] = list()
        self.classifier = classifier
        self._has_ordering = False
    
    def __call__(self, environment: AbstractEnvironment[IT, KT, DT, np.ndarray, RT, LT]) -> LabelEnsemble[IT, KT, DT, RT, LT]:
        super().__call__(environment)
        labelset = self.env.labels.labelset
        self.label_dict = {label: idx for idx, label in enumerate(labelset)}
        self.learners = [
            self._al_builder(self.classifier, label)(environment) for label in labelset
        ]
        return self

    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, np.ndarray, RT, LT]:
        labelcounts = [(self.env.labels.document_count(label), label)
                       for label in self.env.labels.labelset]
        min_label = min(labelcounts)[1]
        al_idx = self.label_dict[min_label]
        learner = self.learners[al_idx]
        return learner
