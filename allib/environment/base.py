from __future__ import annotations

from typing import Generic, Sequence, TypeVar, Any
from abc import ABC, abstractmethod, abstractclassmethod
from ..history import BaseLogger
from instancelib import InstanceProvider, Instance
from instancelib.typehints import KT, DT, VT, RT, LT
from instancelib.labels import LabelProvider

import instancelib as ins

IT = TypeVar("IT", bound="Instance[Any, Any, Any, Any]", covariant=True)

class AbstractEnvironment(ins.AbstractEnvironment[IT, KT, DT, VT, RT, LT], 
                          ABC, Generic[IT, KT, DT, VT, RT, LT]):
    @property
    @abstractmethod
    def unlabeled(self) -> InstanceProvider[IT, KT, DT, VT, RT]:
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
    def labeled(self) -> InstanceProvider[IT, KT, DT, VT, RT]:
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

    @abstractclassmethod
    def from_environment(cls, 
                         environment: AbstractEnvironment[IT, KT, DT, VT, RT, LT],
                         *args, **kwargs
                        ) -> AbstractEnvironment[IT, KT, DT, VT, RT, LT]:
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

    