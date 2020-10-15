from __future__ import annotations

from abc import ABC
from collections import deque
from typing import (Dict, Generic, Iterator, Set,
                    Optional, Sequence,  Tuple, TypeVar, Any)

import pandas as pd # type: ignore

from ..environment import AbstractEnvironment
from ..instances import Instance
from ..machinelearning import AbstractClassifier
from .base import ActiveLearner, NotInitializedException

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")

class PoolbasedAL(ActiveLearner[KT, DT, VT, RT, LT], ABC, Generic[KT, DT, VT, RT, LT, LVT, PVT]):
    def __init__(self,
                 classifier: AbstractClassifier[KT, VT, LT, LVT, PVT],
                 *_, **__
                 ) -> None:
        self.initialized = False
        self._env: Optional[AbstractEnvironment[KT, DT, VT, RT, LT]] = None
        self.classifier = classifier
        self.fitted = False
        self.ordering = None
        self.sampled: Set[KT] = set()

    def __call__(self, 
            environment: AbstractEnvironment[KT, DT, VT, RT, LT]
        ) -> PoolbasedAL[KT, DT, VT, RT, LT, LVT, PVT]:
        self._env = environment
        self._sampled = environment.create_empty_provider()
        self.classifier = self.classifier(environment)
        self.initialized = True
        return self

    @ActiveLearner.iterator_log
    def __next__(self) -> Instance[KT, DT, VT, RT]:
        if self.ordering is None:
            self.ordering = deque(self.calculate_ordering())
        try:
            key = self.ordering.popleft()
            while key not in self.env.unlabeled or key in self.sampled:
                key = self.ordering.popleft()
            self.sampled.add(key)
            return self.env.unlabeled[key]
        except IndexError:
            raise StopIteration()

    @property
    def len_unlabeled(self) -> int:
        """Return the number of unlabeled documents
        
        Returns
        -------
        int
            The number of unlabeled documents
        """ 
        return len(self._unlabeled)

    @property
    def len_labeled(self) -> int:
        return len(self._labeled)

    @property
    def size(self) -> int:
        return self.len_labeled + self.len_unlabeled

    @ActiveLearner.label_log
    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        self.env.unlabeled.discard(instance)
        self.env.labeled.add(instance)

    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as unlabeled
        
        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        self.env.labeled.discard(instance)
        self.env.unlabeled.add(instance)

    def vector_generator(self, only_unlabeled: bool=False) -> Iterator[VT]:
        """Return the vectors of documents contained in this AL Container
        
        Parameters
        ----------
        only_unlabeled : bool, optional
            Do we desire to only generate vectors of unlabeled instances, by default False
        
        Returns
        -------
        Iterator[VT]
            A generator that generates feature vectors

        Yields
        -------
        VT Feature vector
        """        
        if only_unlabeled:
            for doc_id in self.env.unlabeled:
                vector = self.env.unlabeled[doc_id].vector
                if vector is not None:
                    yield vector
        else:
            for _, dat in self.env.dataset.items():
                vector = dat.vector
                if vector is not None:
                    yield vector

    def row_generator(self) -> Iterator[Dict[Any, Any]]:
        """Generate dictionaries that can be used to populate a Pandas DataFrame

        Yields
        -------
        Dict[str, Any]
            A dictionary that contains the necesseary document
            vector properties and label booleans

        """
        for _, doc in self._labeled.items():
            doc_labels = self.env.labels.get_labels(doc)
            label_dict = {
                label: (label in doc_labels) for label in self.env.labels.labelset}
            doc_dict = {
                "id": doc.identifier,
                "vector": doc.vector,
                "data": doc.data
            }
            yield {**doc_dict, **label_dict}

    def xy_generator(self) -> Iterator[Tuple[VT, LVT]]:
        """Generates training examples in vectorized form

        Yields
        -------
        Tuple[VT, np.ndarray]
            A tuple of:
                 x -- the feature vector an instance and 
                 y -- the instances label vector
        """
        for _, doc in self._labeled.items():
            doc_labels = self.env.labels.get_labels(doc)
            x = doc.vector
            y = self.classifier.encode_labels(doc_labels)
            if x is not None and y is not None:
                yield x, y

    def retrain(self) -> None:
        # Ensure instances match labelings
        if not self.initialized:
            raise NotInitializedException
        instances = [instance for _, instance in self._labeled.items()]
        labelings = [self._labelprovider.get_labels(instance) for instance in instances]
        self.classifier.fit_instances(instances, labelings)
        self.fitted = True
        self.ordering = None
        self.sampled = set()

    def predict(self, instances: Sequence[Instance[KT, DT, VT, RT]]):
        if not self.initialized:
            raise NotInitializedException
        return self.classifier.predict_instances(instances)

    def predict_proba(self, instances: Sequence[Instance[KT, DT, VT, RT]]):
        if not self.initialized:
            raise NotInitializedException
        return self.classifier.predict_proba_instances(instances)

    def get_trainingdata(self) -> Optional[pd.DataFrame]:
        """Return all training data from all clusters as a Pandas DataFrame

        Returns
        -------
        Optional[pd.DataFrame]
            The training data if any 
        """
        rows = list(self.row_generator())
        if len(rows) > 0:
            return pd.DataFrame(rows)
        return None
