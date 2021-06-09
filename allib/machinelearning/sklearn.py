from __future__ import annotations

import pickle
import logging
from typing import (Any, FrozenSet, Iterable, List, Sequence, Set, Tuple,
                    TypeVar)

import numpy as np  # type: ignore

from sklearn.base import ClassifierMixin, TransformerMixin  # type: ignore

from ..balancing import BaseBalancer, IdentityBalancer
from ..environment import AbstractEnvironment
from instancelib.instances import Instance
from ..utils import SaveableInnerModel
from ..utils.func import list_unzip
from .base import AbstractClassifier

LOGGER = logging.getLogger(__name__)

class SkLearnClassifier(SaveableInnerModel, AbstractClassifier[int, np.ndarray, str, np.ndarray, np.ndarray]):
    _name = "Sklearn"

    def __init__(
            self,
            estimator: ClassifierMixin, encoder: TransformerMixin, balancer: BaseBalancer = IdentityBalancer(),
            storage_location=None, filename=None) -> None:
        SaveableInnerModel.__init__(self, estimator, storage_location, filename)
        self.encoder = encoder 
        self._fitted = False
        self._target_labels: FrozenSet[str] = frozenset()
        self.balancer = balancer

    def __call__(self, environment: AbstractEnvironment[Any, int, Any, np.ndarray, Any, str]) -> SkLearnClassifier:
        self._target_labels = frozenset(environment.labels.labelset)
        self.encoder.fit(list(self._target_labels)) # type: ignore
        return self

    def encode_labels(self, labels: Iterable[str]) -> np.ndarray:
        return self.encoder.transform(list(set(labels))) # type: ignore

    def decode_vector(self, vector: np.ndarray) -> Sequence[FrozenSet[str]]:
        labelings = self.encoder.inverse_transform(vector).tolist() # type: ignore
        return [frozenset([labeling]) for labeling in labelings]

    def get_label_column_index(self, label: str) -> int:
        label_list = self.encoder.classes_.tolist() # type: ignore
        return label_list.index(label)

    @SaveableInnerModel.load_model_fallback
    def fit(self, x_data: np.ndarray, y_data: np.ndarray):
        assert x_data.shape[0] == y_data.shape[0]
        x_resampled, y_resampled = self.balancer.resample(x_data, y_data)
        LOGGER.info("[%s] Balanced / Resampled the data", self.name)
        self.innermodel.fit(x_resampled, y_resampled) # type: ignore
        LOGGER.info("[%s] Fitted the model", self.name)
        self._fitted = True

    def encode_xy(self, instances: Sequence[Instance[int, Any, np.ndarray, Any]], labelings: Sequence[Iterable[str]]):
        def yield_xy():
            for ins, lbl in zip(instances, labelings):
                if ins.vector is not None:
                    yield ins.vector, self.encode_labels(lbl)
        x_data, y_data = list_unzip(yield_xy())
        x_fm = np.vstack(x_data)
        y_lm = np.vstack(y_data)
        if y_lm.shape[1] == 1:
            y_lm = np.reshape(y_lm, (y_lm.shape[0],))
        return x_fm, y_lm

    def encode_x(self, instances: Sequence[Instance[int, Any, np.ndarray, Any]]) -> np.ndarray:
        # TODO Maybe convert to staticmethod
        x_data = [
            instance.vector for instance in instances if instance.vector is not None]
        x_vec = np.vstack(x_data)
        return x_vec

    def encode_y(self, labelings: Sequence[Iterable[str]]) -> np.ndarray:
        y_data = [self.encode_labels(labeling) for labeling in labelings]
        y_vec = np.vstack(y_data)
        if y_vec.shape[1] == 1:
            y_vec = np.reshape(y_vec, (y_vec.shape[0],))
        return y_vec

    @SaveableInnerModel.load_model_fallback
    def predict_proba(self, x_data: np.ndarray) -> np.ndarray:
        assert self.innermodel is not None
        return self.innermodel.predict_proba(x_data) 

    @SaveableInnerModel.load_model_fallback
    def predict(self, x_data: np.ndarray) -> np.ndarray:
        assert self.innermodel is not None
        return self.innermodel.predict(x_data)

    def predict_instances(self, instances: Sequence[Instance[int, Any, np.ndarray, Any]]) -> Sequence[FrozenSet[str]]:
        x_vec = self.encode_x(instances)
        y_pred = self.predict(x_vec)
        return self.decode_vector(y_pred)

    def predict_proba_instances(self, instances: Sequence[Instance[int, Any, np.ndarray, Any]]) -> Sequence[FrozenSet[Tuple[str, float]]]:
        x_vec = self.encode_x(instances)
        y_pred = self.predict_proba(x_vec).tolist()
        label_list: List[str] = self.encoder.classes_.tolist() # type: ignore
        y_labels: List[FrozenSet[Tuple[str, float]]] = [
            frozenset(zip(label_list, y_vec)) # type: ignore
            for y_vec in y_pred
        ]
        return y_labels
    
    @property
    def name(self) -> str:
        return f"{self._name} :: {self.innermodel.__class__}"
        
    def fit_vectors(self, x_data: Sequence[np.ndarray], labels: Sequence[FrozenSet[str]]):
        x_mat = np.vstack(x_data)
        y_vec = self.encode_y(labels)
        self.fit(x_mat, y_vec)

    def fit_instances(self, instances: Sequence[Instance[int, Any, np.ndarray, Any]], labels: Sequence[FrozenSet[str]]):
        assert len(instances) == len(labels)
        x_train_vec, y_train_vec = self.encode_xy(instances, labels)
        self.fit(x_train_vec, y_train_vec)

    @property
    def fitted(self) -> bool:
        return self._fitted




class MultilabelSkLearnClassifier(SkLearnClassifier):
    _name = "Multilabel Sklearn"
    def __call__(self, environment: AbstractEnvironment[Any, int, Any, np.ndarray, Any, str]) -> SkLearnClassifier:
        self._target_labels = frozenset(environment.labels.labelset)
        self.encoder.fit(list(map(lambda x: {x}, self._target_labels))) # type: ignore
        return self

    def encode_labels(self, labels: Iterable[str]) -> np.ndarray:
        return self.encoder.transform([list(set(labels))]) # type: ignore

    def decode_vector(self, vector: np.ndarray) -> Sequence[FrozenSet[str]]:
        labelings = self.encoder.inverse_transform(vector) # type: ignore
        return [frozenset(labeling) for labeling in labelings]
