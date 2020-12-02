from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (Any, Callable, Dict, Generic, Iterator, List, Optional, Sequence,
                    Tuple, TypeVar)

import numpy as np  # type: ignore

from ..environment import AbstractEnvironment
from ..instances.base import Instance
from ..machinelearning import AbstractClassifier
from .base import ActiveLearner
from .ml_based import MLBased, ProbabiltyBased, FeatureMatrix
from .ensembles import AbstractEnsemble

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")
FT = TypeVar("FT")



class LabelProbabilityBased(ProbabiltyBased[KT, DT, RT, LT], ABC, Generic[KT, DT, RT, LT]):
    def __init__(self, classifier: AbstractClassifier[KT, VT, LT, LVT, PVT], label: LT, *_, **__) -> None:
        super().__init__(classifier)
        self.label = label
        self.labelposition: Optional[int] = None

    def __call__(self, environment: AbstractEnvironment[KT, DT, np.ndarray, RT, LT]) -> LabelProbabilityBased[KT, DT, RT, LT]:
        super().__call__(environment)
        self.labelposition = self.classifier.get_label_column_index(self.label)
        return self

    @property
    def name(self) -> Tuple[str, LT]:
        return f"{self._name} :: {self.classifier.name}", self.label

    @staticmethod
    @abstractmethod
    def selection_criterion(prob_vec: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _get_predictions(self, matrix: FeatureMatrix[KT]) -> Tuple[Sequence[KT], np.ndarray]:
        prob_vec: np.ndarray = self.classifier.predict_proba(
            matrix.matrix)  # type: ignore
        # type: ignore
        sliced_prob_vec: np.ndarray = prob_vec[:, self.labelposition] # type: ignore
        keys = matrix.indices
        return keys, sliced_prob_vec


class LabelEnsemble(AbstractEnsemble[KT, DT, np.ndarray, RT, LT], MLBased[KT,DT, np.ndarray, RT, LT, np.ndarray, np.ndarray],Generic[KT, DT, RT, LT]):
    _name = "LabelEnsemble"

    def __init__(self,
                 classifier: AbstractClassifier[KT, np.ndarray, LT, np.ndarray, np.ndarray],
                 al_method: Callable[..., LabelProbabilityBased[KT, DT, RT, LT]],
                 *_, **__
                 ) -> None:
        self._al_builder = al_method
        self.label_dict: Dict[int, LT] = dict()
        self.learners: List[LabelProbabilityBased[KT, DT, RT, LT]] = list()
        self.classifier = classifier

    def __call__(self, environment: AbstractEnvironment[KT, DT, np.ndarray, RT, LT]) -> LabelEnsemble[KT, DT, RT, LT]:
        super().__call__(environment)
        labelset = self.env.labels.labelset
        self.label_dict = dict(enumerate(labelset))
        self.learners = [
            self._al_builder(self.classifier, label)(environment) for label in labelset
        ]
        return self

    def _choose_learner(self) -> ActiveLearner[KT, DT, VT, RT, LT]:
        labelcounts = [(self.env.labels.document_count(label), label)
                       for label in self.env.labels.labelset]
        min_label = min(labelcounts)[1]
        learner = self.learners[min_label]
        return learner