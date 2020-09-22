from __future__ import annotations
import functools
import logging
import sys

from abc import ABC, abstractmethod
from typing import (Callable, Dict, Generic, Iterable, List, Iterator,
                    Optional, Sequence, Tuple, TypeVar, Union, Deque)
from collections import deque

from environment import AbstractEnvironment
from instances import Instance, InstanceProvider

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
LVT = TypeVar("LVT")

BasePrediction = List[Tuple[LT, float]]
ChildPrediction = Dict[KT, BasePrediction]
Prediction = Union[BasePrediction, ChildPrediction]

LOGGER = logging.getLogger(__name__)

class NotInitializedException(Exception):
    pass


class ActiveLearner(ABC, Iterator[Instance], Generic[KT, LT]):
    _name = "ActiveLearner"
    ordering: Deque[KT]
    _unlabeled: InstanceProvider
    _labeled: InstanceProvider

    @property
    def name(self):
        return self._name
   
    def __iter__(self) -> ActiveLearner:
        return self

    @abstractmethod
    def calculate_ordering(self) -> List[KT]:
        raise NotImplementedError
    
    @abstractmethod
    def __next__(self) -> Instance:
        raise NotImplementedError
       
    @staticmethod
    def iterator_log(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            LOGGER.info("Sampled document %i with method %s",
                            result.identifier, self.name)
            return result
        return wrapper

    @staticmethod
    def query_log(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result = func(self, *args, **kwargs)
            if "disable_logging" not in kwargs or not kwargs["disable_logging"]:
                LOGGER.info("Sampled document %i with method %s",
                            result.identifier, self.name)
            return result
        return wrapper

    @staticmethod
    def query_batch_log(func):
        @functools.wraps(func)
        def wrapper(self, *args, **kwargs):
            result_batch = func(self, *args, **kwargs)
            if "disable_logging" not in kwargs or not kwargs["disable_logging"]:
                for instance in result_batch:
                    LOGGER.info("Sampled document %i with method %s",
                                instance.identifier, self.name)
                    print(
                        f"Sampled document {instance.identifier} with method {self.name}")
            return result_batch
        return wrapper

    @abstractmethod
    def __call__(self, environment: AbstractEnvironment) -> ActiveLearner:
        raise NotImplementedError

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

    @abstractmethod
    def set_as_labeled(self, instance: Instance) -> None:
        """Mark the instance as labeled

        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        raise NotImplementedError

    @abstractmethod
    def set_as_sampled(self, instance: Instance) -> None:
        """Mark the instance as labeled

        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        raise NotImplementedError

    @abstractmethod
    def set_as_unlabeled(self, instance: Instance) -> None:
        """Mark the instance as unlabeled

        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        raise NotImplementedError

    @abstractmethod
    def retrain(self) -> None:
        """Retrain the model based on the current information
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, instances: Sequence[Instance]) -> Sequence[Prediction]:
        """Return the labeling of the instance

        Parameters
        ----------
        instance : Instance
            The Instance

        Returns
        -------
        Prediction
            The prediction for this instance
        """
        raise NotImplementedError

    @abstractmethod
    def predict_proba(self, instances: Sequence[Instance]):
        """Return the labeling of the instance

        Parameters
        ----------
        instance : Instance
            The Instance

        Returns
        -------
        Prediction
            The prediction for this instance
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def len_unlabeled(self) -> int:
        """Return the number of unlabeled documents

        Returns
        -------
        int
            The number of labeled documents
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def len_labeled(self) -> int:
        """Return the number of labeled documents

        Returns
        -------
        int
            The number of labeled documents
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def size(self) -> int:
        """The number of initial unlabeled documents

        Returns
        -------
        int
            The number of unlabeled documents
        """
        raise NotImplementedError

    @property
    def ratio_learned(self) -> float:
        """The labeling progress sofar

        Returns
        -------
        float
            the ratio
        """
        return self.len_labeled / self.size

    @staticmethod
    def ordered_max(instances: InstanceProvider, fun: Callable[[VT], float]) -> List[KT]:
        funmap = ((key, fun(instances[key].vector)) for key in instances)
        return sorted(funmap, key=lambda x: x[1], reverse=True)

    @staticmethod
    def argmax_n(instances: InstanceProvider,
                 fun: Callable[[VT], float],
                 count: int) -> List[KT]:
        return [t[0] for t in ActiveLearner.ordered_max(instances, fun)[:count]]

    @staticmethod
    def argmin_n(instances: InstanceProvider,
                 fun: Callable[[VT], float],
                 count: int) -> List[KT]:
        return [t[0] for t in ActiveLearner.ordered_max(instances, fun)[-count:]]

    @staticmethod
    def argmax(instances: InstanceProvider, fun: Callable[[VT], float]) -> KT:
        return ActiveLearner.ordered_max(instances, fun)[0][0]

    @staticmethod
    def argmin(instances: InstanceProvider, fun: Callable[[VT], float]) -> KT:
        return ActiveLearner.ordered_max(instances, fun)[-1][0]
