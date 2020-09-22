from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (Callable, Dict, Generic, Iterable, Iterator, List,
                    Optional, Sequence, Set, Tuple, TypeVar, Union)
from collections import deque
import pandas as pd
from environment import AbstractEnvironment
from instances import Instance, InstanceProvider
from labels import LabelProvider
from machinelearning import AbstractClassifier
from .base import ActiveLearner, NotInitializedException

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
LVT = TypeVar("LVT")

BasePrediction = List[Tuple[LT, float]]
ChildPrediction = Dict[KT, BasePrediction]
Prediction = Union[BasePrediction, ChildPrediction]


class PoolbasedAL(ActiveLearner, ABC, Generic[KT, VT, DT, LT, LVT]):
    def __init__(self,
                 classifier: AbstractClassifier
                 ) -> None:
        self.initialized = False
        self._labelprovider = None
        self._whole = None
        self._unlabeled = None
        self._labeled = None
        self._sampled = None
        self._env = None
        self.classifier = classifier
        self.fitted = False
        self.ordering = None

    def __call__(self, environment: AbstractEnvironment) -> PoolbasedAL:
        self._env = environment
        self._labelprovider = environment.label_provider
        self._whole = environment.dataset_provider
        self._unlabeled = environment.unlabeled_provider
        self._labeled = environment.labeled_provider
        self._sampled = environment.create_empty_provider()
        self.classifier = self.classifier(environment)
        self.initialized = True
        return self

    @ActiveLearner.iterator_log
    def __next__(self) -> Instance:
        if self.ordering is None:
            self.ordering = deque(self.calculate_ordering())
        try:
            key = self.ordering.popleft()
            while key not in self._unlabeled:
                key = self.ordering.popleft()
            return self._unlabeled[key]
        except IndexError:
            raise StopIteration()

    @abstractmethod
    def query(self) -> Optional[Instance]:
        """Query the most informative instance
        Returns
        -------
        Optional[Instance]
            The most informative instance
        """        
        raise NotImplementedError
    
    @abstractmethod
    def query_batch(self, batch_size: int) -> List[Instance]:
        """Query the `batch_size` most informative instances
        
        Parameters
        ----------
        batch_size : int
            The size of the batch
        
        Returns
        -------
        List[Instance]
            A batch with `len(batch) <= batch_size` 
        """        
        raise NotImplementedError

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

    def set_as_labeled(self, instance: Instance) -> None:
        self._unlabeled.discard(instance)
        self._labeled.add(instance)

    def set_as_sampled(self, instance: Instance) -> None:
        """Mark the instance as labeled
        
        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        self._unlabeled.discard(instance)
        self._sampled.add(instance)

    def set_as_unlabeled(self, instance: Instance) -> None:
        """Mark the instance as unlabeled
        
        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        self._sampled.discard(instance)
        self._labeled.discard(instance)
        self._unlabeled.add(instance)

    def vector_generator(self, only_unlabeled=False) -> Iterator[VT]:
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
            for doc_id in self._unlabeled:
                yield self._unlabeled[doc_id].vector
        else:
            for _, dat in self._whole.items():
                yield dat.vector

    def row_generator(self) -> Iterator[Dict[str, Union[KT, VT, DT, LT]]]:
        """Generate dictionaries that can be used to populate a Pandas DataFrame

        Yields
        -------
        Dict[str, Any]
            A dictionary that contains the necesseary document
            vector properties and label booleans

        """
        for _, doc in self._labeled.items():
            doc_labels = self._labelprovider.get_labels(doc)
            label_dict = {
                label: (label in doc_labels) for label in self._labelprovider.labelset}
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
            doc_labels = self._labelprovider.get_labels(doc)
            x = doc.vector
            y = self.classifier.encode_labels(doc_labels)
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

    def predict(self, instances: Sequence[Instance]):
        if not self.initialized:
            raise NotInitializedException
        return self.classifier.predict_instances(instances)

    def predict_proba(self, instances: Sequence[Instance]):
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
