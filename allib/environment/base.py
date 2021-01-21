from __future__ import annotations

from typing import Generic, Sequence, TypeVar, Any
from abc import ABC, abstractmethod, abstractclassmethod
from ..history import BaseLogger
from ..instances import InstanceProvider
from ..labels import LabelProvider

KT = TypeVar("KT")
LT = TypeVar("LT")
VT = TypeVar("VT")
DT = TypeVar("DT")
RT = TypeVar("RT")

class AbstractEnvironment(ABC, Generic[KT, DT, VT, RT, LT]):
    @abstractmethod
    def create_empty_provider(self) -> InstanceProvider[KT, DT, VT, RT]:
        """Use this method to create an empty `InstanceProvider`

        Returns
        -------
        InstanceProvider[KT, DT, VT, RT]
            The newly created provider
        """        
        raise NotImplementedError

    @property
    @abstractmethod
    def dataset(self) -> InstanceProvider[KT, DT, VT, RT]:
        """This property contains the `InstanceProvider` that contains
        the whole dataset. This provider should include all instances
        that are contained in the other providers.

        Returns
        -------
        InstanceProvider[KT, DT, VT, RT]
            The dataset `InstanceProvider`
        """        
        raise NotImplementedError

    @property
    @abstractmethod
    def unlabeled(self) -> InstanceProvider[KT, DT, VT, RT]:
        """This `InstanceProvider` contains all unlabeled instances.
        `ActiveLearner` methods sample instances from this provider/

        Returns
        -------
        InstanceProvider[KT, DT, VT, RT]
            An `InstanceProvider` that contains all unlabeld instances
        """        
        raise NotImplementedError

    @property
    @abstractmethod
    def labeled(self) -> InstanceProvider[KT, DT, VT, RT]:
        """This `InstanceProvider` contains all labeled instances.
        `ActiveLearner` may use this provider to train a classifier
        to sample instances from the `unlabeled` provider.

        Returns
        -------
        InstanceProvider[KT, DT, VT, RT]
            An `InstanceProvider` that contains all labeled instances
        """        
        raise NotImplementedError

    @property
    @abstractmethod
    def logger(self) -> BaseLogger[KT, LT, Any]:
        """This property contains an implementation of a `BaseLogger`.
        The logger can log events like sampling an instance, or its labeling.

        Returns
        -------
        BaseLogger[KT, LT, Any]
            The logger
        """        
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self) -> LabelProvider[KT, LT]:
        """This property contains provider that maps instances to labels and
        vice-versa. 

        Returns
        -------
        LabelProvider[KT, LT]
            The label provider
        """        
        raise NotImplementedError

    @property
    @abstractmethod
    def truth(self) -> LabelProvider[KT, LT]:
        """This property contains a `LabelProvider` that maps 
        instances to *ground truth* labels and vice-versa. 
        This can be used for simulation purposes if you want
        to assess the performance of an AL algorithm on a dataset
        with a ground truth.

        Returns
        -------
        LabelProvider[KT, LT]
            The label provider that contains the ground truth labels
        """
        raise NotImplementedError

    @abstractmethod
    def store(self, key: str, value: Any) -> None:
        """Use this method to store some data in this environment's 
        Key Value storage that can be retrieved later. 

        Parameters
        ----------
        key : str
            An identification key for the data
        value : Any
            The data that should be stored
        """        
        raise NotImplementedError

    @abstractmethod
    def storage_exists(self, key: str) -> bool:
        """Check if documents with `key` exist in this Environment/

        Parameters
        ----------
        key : str
            The identification key

        Returns
        -------
        bool
            True if there is data stored for `key`
        """        
        raise NotImplementedError

    @abstractmethod
    def retrieve(self, key: str) -> Any:
        """Retrieve the data stored with key `key`.

        Parameters
        ----------
        key : str
            The identification key

        Returns
        -------
        Any
            The data stored for key `key`
        """        
        raise NotImplementedError

    def add_vectors(self, keys: Sequence[KT], vectors: Sequence[VT]) -> None:
        """This method adds feature vectors or embeddings to instances 
        associated with the keys in the first parameters. The sequences
        `keys` and `vectors` should have the same length.


        Parameters
        ----------
        keys : Sequence[KT]
            A sequence of keys
        vectors : Sequence[VT]
            A sequence of vectors that should be associated with the instances 
            of the sequence `keys`
        """        
        self.dataset.bulk_add_vectors(keys, vectors)

    @abstractclassmethod
    def from_environment(cls, 
                         environment: AbstractEnvironment[KT, DT, VT, RT, LT],
                         *args, **kwargs
                        ) -> AbstractEnvironment[KT, DT, VT, RT, LT]:
        """Create a new independent environment with the same state.
        Implementations may enable conversion from and to several types
        of Enviroments.

        Parameters
        ----------
        environment : AbstractEnvironment[KT, DT, VT, RT, LT]
            The environment that should be duplicated

        Returns
        -------
        AbstractEnvironment[KT, DT, VT, RT, LT]
            A new independent with the same state
        """        
        raise NotImplementedError

