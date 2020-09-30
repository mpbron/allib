from abc import ABC, abstractmethod, abstractclassmethod

from typing import Iterator, Generic, TypeVar, Union, Iterable, Set, FrozenSet

from ..instances import Instance, InstanceProvider

LT = TypeVar("LT")
KT = TypeVar("KT")


def to_key(instance_or_key: Union[KT, Instance]) -> KT:
    """Returns the identifier of the instance if `instance_or_key` is an `Instance`
    or return the key if `instance_or_key` is a `KT`

    Parameters
    ----------
    instance_or_key : Union[KT, Instance]
        An implementation of `Instance` or an identifier typed variable

    Returns
    -------
    KT
        The identifer of the instance (or the input verbatim)
    """
    if isinstance(instance_or_key, Instance):
        return instance_or_key.identifier
    return instance_or_key


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
    def remove_labels(self, instance: Union[KT, Instance], *labels: LT) -> None:
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
    def set_labels(self, instance: Union[KT, Instance], *labels: LT) -> None:
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
    def get_labels(self, instance: Union[KT, Instance]) -> Set[LT]:
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
