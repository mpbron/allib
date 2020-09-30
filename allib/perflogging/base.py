from typing import Generic, List, Sequence, TypeVar, Tuple

from abc import ABC, abstractmethod
from ..instances import Instance

KT = TypeVar("KT")
LT = TypeVar("LT")
MT = TypeVar("MT")
ST = TypeVar("ST")



class BaseLogger(ABC, Generic[LT, KT, ST]):
    @abstractmethod
    def set_labels(self, x: Instance, y: str) -> None:
        """Add/set labels to state

        If the labels do not exist, add it to the state.

        Arguments
        ---------
        y: np.array
            One dimensional integer numpy array with inclusion labels.
        """
        raise NotImplementedError

       
    @abstractmethod
    def get_current_queries(self):
        """Get the current queries made by the model.

        This is useful to get back exactly to the state it was in before
        shutting down a review.

        Returns
        -------
        dict:
            The last known queries according to the state file.
        """
        raise NotImplementedError

    @abstractmethod
    def set_current_queries(self, current_queries: Sequence[Tuple[Instance, List[ST]]]):
        """Set the current queries made by the model.

        Arguments
        ---------
        current_queries: dict
            The last known queries, with [(query_idx, query_method)].
        """
        raise NotImplementedError

    @abstractmethod
    def add_proba(self, pool_idx, train_idx, proba, query_i):
        """Add inverse pool indices and their labels.

        Arguments
        ---------
        indices: list, np.array
            A list of indices used for unlabeled pool.
        pred: np.array
            Array of prediction probabilities for unlabeled pool.
        i: int
            The query number.
        """
        raise NotImplementedError

    
    @property
    @abstractmethod
    def n_queries(self):
        """Number of queries saved in the logger.

        Returns
        -------
        int:
            Number of queries.
        """
        raise NotImplementedError

    @abstractmethod
    def store_sample_info(self, instance: Instance, sample_methods: List[ST]):
        raise NotImplementedError

    @abstractmethod
    def store_metric(self, key: str, metric: MT, value: float) -> None:
        raise NotImplementedError

    @abstractmethod
    def store_info_score(self, instance: Instance, score: float) -> None:
        raise NotImplementedError

    
    def store_probabilities(self, instance: Instance, label: LT, score: float) -> None: 
        raise NotImplementedError

 
