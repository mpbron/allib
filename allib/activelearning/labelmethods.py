from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)

import numpy.typing as npt

from ..environment.base import AbstractEnvironment
from ..machinelearning.base import AbstractClassifier
from ..typehints import DT, IT, KT, LT, RT
from .base import ActiveLearner
from .ensembles import AbstractEnsemble
from .selectioncriterion import AbstractSelectionCriterion
from .ml_based import FeatureMatrix, MLBased, ProbabilityBased
from .random import RandomSampling


class LabelProbabilityBased(
    ProbabilityBased[IT, KT, DT, RT, LT], ABC, Generic[IT, KT, DT, RT, LT]
):
    def __init__(
        self,
        env: AbstractEnvironment[IT, KT, DT, npt.NDArray[Any], RT, LT],
        classifier: AbstractClassifier[
            KT, npt.NDArray[Any], LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        selection_criterion: AbstractSelectionCriterion,
        label: LT,
        fallback: ActiveLearner[IT, KT, DT, npt.NDArray[Any], RT, LT],
        identifier: Optional[str] = None,
        *_,
        **__,
    ) -> None:
        super().__init__(
            env, classifier, selection_criterion, fallback, identifier=identifier
        )
        self.label = label
        self.labelposition: int = self.classifier.get_label_column_index(self.label)

    @property
    def name(self) -> Tuple[str, LT]:
        if self.identifier is not None:
            return f"{self.identifier}", self.label
        return f"{self._name} :: {self.classifier.name}", self.label

    def _get_predictions(
        self, matrix: FeatureMatrix[KT]
    ) -> Tuple[Sequence[KT], npt.NDArray[Any]]:
        prob_vec: npt.NDArray[Any] = self.classifier.predict_proba(
            matrix.matrix
        )  # type: ignore
        # type: ignore
        sliced_prob_vec: npt.NDArray[Any] = prob_vec[:, self.labelposition]  # type: ignore
        keys = matrix.indices
        return keys, sliced_prob_vec

    @classmethod
    def builder(
        cls,
        classifier: AbstractClassifier[
            KT, npt.NDArray[Any], LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        selection_criterion: AbstractSelectionCriterion,
        fallback_builder: Callable[
            ..., ActiveLearner[IT, KT, DT, npt.NDArray[Any], RT, LT]
        ],
        batch_size=200,
        *_,
        identifier: Optional[str] = None,
        **__,
    ) -> Callable[..., ActiveLearner[IT, KT, DT, npt.NDArray[Any], RT, LT],]:
        def wrap_func(
            env: AbstractEnvironment[IT, KT, DT, npt.NDArray[Any], RT, LT], 
            pos_label: LT, *_, **__
        ):
            built_classifier = classifier(env)
            fallback = fallback_builder(env)
            return cls(
                env,
                built_classifier,
                selection_criterion,
                pos_label,
                fallback,
                batch_size=batch_size,
                identifier=identifier,
            )
        return wrap_func


class LabelEnsemble(
    AbstractEnsemble[IT, KT, DT, npt.NDArray[Any], RT, LT],
    MLBased[IT, KT, DT, npt.NDArray[Any], RT, LT, npt.NDArray[Any], npt.NDArray[Any]],
    Generic[IT, KT, DT, RT, LT],
):
    _name = "LabelEnsemble"

    def __init__(
        self,
        classifier: AbstractClassifier[
            KT, npt.NDArray[Any], LT, npt.NDArray[Any], npt.NDArray[Any]
        ],
        al_method: Callable[..., LabelProbabilityBased[IT, KT, DT, RT, LT]],
        *_,
        **__,
    ) -> None:
        self._al_builder = al_method
        self.label_dict: Dict[LT, int] = dict()
        self.learners: List[
            ActiveLearner[IT, KT, DT, npt.NDArray[Any], RT, LT]
        ] = list()
        self.classifier = classifier
        self._has_ordering = False

    def __call__(
        self, environment: AbstractEnvironment[IT, KT, DT, npt.NDArray[Any], RT, LT]
    ) -> LabelEnsemble[IT, KT, DT, RT, LT]:
        super().__call__(environment)
        labelset = self.env.labels.labelset
        self.label_dict = {label: idx for idx, label in enumerate(labelset)}
        self.learners = [
            self._al_builder(self.classifier, label)(environment) for label in labelset
        ]
        return self

    def _choose_learner(self) -> ActiveLearner[IT, KT, DT, npt.NDArray[Any], RT, LT]:
        labelcounts = [
            (self.env.labels.document_count(label), label)
            for label in self.env.labels.labelset
        ]
        min_label = min(labelcounts)[1]
        al_idx = self.label_dict[min_label]
        learner = self.learners[al_idx]
        return learner
