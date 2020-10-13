from __future__ import annotations

from allib.labels.base import LabelProvider
import functools
import itertools
import logging


from abc import ABC, abstractmethod
from typing import (Callable, Dict, FrozenSet, Generic, List, Iterator,
                    Optional, Sequence, Tuple, TypeVar, Deque, Any, Union)

from ..environment import AbstractEnvironment
from ..instances import Instance, InstanceProvider

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
FT = TypeVar("FT")
F = TypeVar("F", bound=Callable[..., Any])

ProbabilityPrediction = FrozenSet[Tuple[LT, float]]
LabelPrediction = FrozenSet[LT]

LOGGER = logging.getLogger(__name__)

class NotInitializedException(Exception):
    pass


class ActiveLearner(ABC, Iterator[Instance[KT, DT, VT, RT]], Generic[KT, DT, VT, RT, LT]):
    _name = "ActiveLearner"
    ordering: Optional[Deque[KT]]
    _env: Optional[AbstractEnvironment[KT, DT, VT, RT, LT]]

    @property
    def name(self) -> Tuple[str, Optional[LT]]:
        return self._name, None
   
    def __iter__(self) -> ActiveLearner[KT, DT, VT, RT, LT]:
        return self

    @property
    def env(self) -> AbstractEnvironment[KT, DT, VT, RT, LT]:
        if self._env is None:
            raise NotInitializedException
        return self._env

    @property
    def _unlabeled(self) -> InstanceProvider[KT, DT, VT, RT]:
        return self.env.unlabeled
    
    @property
    def _labeled(self) -> InstanceProvider[KT, DT, VT, RT]:
        return self.env.labeled
    
    @property
    def _dataset(self) -> InstanceProvider[KT, DT, VT, RT]:
        return self.env.dataset

    @property
    def _labelprovider(self) -> LabelProvider[KT, LT]:
        return self.env.labels
    
    @abstractmethod
    def calculate_ordering(self) -> Sequence[KT]:
        raise NotImplementedError
    
    @abstractmethod
    def __next__(self) -> Instance[KT, DT, VT, RT]:
        raise NotImplementedError
       
    @staticmethod
    def iterator_log(func: F) -> F:
        @functools.wraps(func)
        def wrapper(
                self: ActiveLearner[KT, DT, VT, RT, LT], 
                *args: Any, **kwargs: Dict[str, Any]) -> FT:
            result: Union[Any,Instance[KT, DT, VT, RT]] = func(self, *args, **kwargs)
            if isinstance(result, Instance):
                LOGGER.info("Sampled document %i with method %s",
                            result.identifier, self.name) # type: ignore
                self.env.logger.log_sample(result, self.name)
            return result # type: ignore
        return wrapper
    
    @staticmethod
    def label_log(func: F) -> F:
        @functools.wraps(func)
        def wrapper(
            self: ActiveLearner[KT, DT, VT, RT, LT], 
            instance: Instance[KT, DT, VT, RT], *args: Any, **kwargs: Any):
            labels = self.env.labels.get_labels(instance)
            self.env.logger.log_label(instance, self.name,  *labels)
            return func(self, instance, *args, **kwargs)
        return wrapper

    @abstractmethod
    def __call__(
            self, 
            environment: AbstractEnvironment[KT, DT, VT, RT, LT]) -> ActiveLearner[KT, DT, VT, RT, LT]:
        raise NotImplementedError

    def query(self) -> Optional[Instance[KT, DT, VT, RT]]:
        """Query the most informative instance
        Returns
        -------
        Optional[Instance]
            The most informative instance
        """
        return next(self, None)

   
    def query_batch(self, batch_size: int) -> List[Instance[KT, DT, VT, RT]]:
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
        return list(itertools.islice(self, batch_size))

    @abstractmethod
    def set_as_labeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Mark the instance as labeled

        Parameters
        ----------
        instance : Instance
            The now labeled instance
        """
        raise NotImplementedError

    @abstractmethod
    def set_as_unlabeled(self, instance: Instance[KT, DT, VT, RT]) -> None:
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
    def predict(self, instances: Sequence[Instance[KT, DT, VT, RT]]) -> Sequence[LabelPrediction]:
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
    def predict_proba(self, instances: Sequence[Instance[KT, DT, VT, RT]]) -> Sequence[ProbabilityPrediction]:
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
