from abc import ABC, abstractmethod
from typing import Any, FrozenSet, Generic, Set, TypeVar, Union

from ..instances import Instance
from ..utils.to_key import to_key

LT = TypeVar("LT")
KT = TypeVar("KT")


class LabelProvider(ABC, Generic[KT, LT]):
    @property
    @abstractmethod
    def labelset(self) -> FrozenSet[LT]:
        """Report all possible labels (example usage: for setting up a classifier)

        Returns
        -------
        Set[LT]
            Labels of type `LT`
        """
        raise NotImplementedError

    @abstractmethod
    def remove_labels(self, instance: Union[KT, Instance[KT, Any, Any, Any]], *labels: LT) -> None:
        """Remove the labels from this instance

        Parameters
        ----------
        instance : Union[KT, Instance]
            The instance
        *labels: LT
            The labels that should be removed from the instance
        """
        raise NotImplementedError

    @abstractmethod
    def set_labels(self, instance: Union[KT, Instance[KT, Any, Any, Any]], *labels: LT) -> None:
        """Annotate the instance with the labels listed in the parameters

        Parameters
        ----------
        instance : Union[KT, Instance]
            The instance
        *labels: LT
            The labels that should be associated with the instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_labels(self, instance: Union[KT, Instance[KT, Any, Any, Any]]) -> Set[LT]:
        """Return the labels that are associated with the instance

        Parameters
        ----------
        instance : Union[KT, Instance]
            The instance

        Returns
        -------
        Set[LT]
            The labels that are associated with the instance
        """
        raise NotImplementedError

    @abstractmethod
    def get_instances_by_label(self, label: LT) -> Set[KT]:
        """Retrieve which instances are annotated with `label`

        Parameters
        ----------
        label : LT
            A Label

        Returns
        -------
        Set[Instance]
            The identifiers of the instance 
        """
        raise NotImplementedError

    @property
    def len_positive(self) -> int:
        docset = set()
        for label in self.labelset:
            for instance in self.get_instances_by_label(label):
                docset.add(instance)
        return len(docset)

    def document_count(self, label: LT) -> int:
        return len(self.get_instances_by_label(label))
