from __future__ import annotations

from abc import ABC, abstractmethod
from ..utils.func import filter_snd_none, filter_snd_none_zipped
from ..utils.chunks import divide_iterable_in_lists
from typing import Generic, Iterator, Sequence, List, Optional, TypeVar, Any, Mapping, MutableMapping, Tuple

import numpy as np #type: ignore

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")
CT = TypeVar("CT")
LVT = TypeVar("LVT")

class Instance(ABC, Generic[KT, DT, VT, RT]):

    @property
    @abstractmethod
    def data(self) -> DT:
        """Return the raw data of this instance


        Returns
        -------
        DT
            The Raw Data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def representation(self) -> RT:
        """Return a representation for annotation


        Returns
        -------
        RT
            A representation of the raw data
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def vector(self) -> Optional[VT]:
        """Get the vector represenation of the raw data

        Returns
        -------
        Optional[VT]
            The Vector
        """
        raise NotImplementedError

    @vector.setter
    def vector(self, value: Optional[VT]) -> None: # type: ignore
        raise NotImplementedError

    @property
    @abstractmethod
    def identifier(self) -> KT:
        """Get the identifier of the instance

        Returns
        -------
        KT
            The identifier key of the instance
        """
        raise NotImplementedError

    def __str__(self) -> str:
        str_rep = f"Instance:\n Identifier => {self.identifier} \n Data => {self.data} \n Vector present => {self.vector is not None}"
        return str_rep

    def __repr__(self) -> str:
        return self.__str__()


class ContextInstance(Instance[KT, DT, VT, RT], ABC, Generic[KT, DT, VT, RT, CT]):
    @property
    @abstractmethod
    def context(self) -> CT:
        """Get the context of this instance

        Returns
        -------
        CT
            The context of this instance
        """        
        raise NotImplementedError


class ChildInstance(Instance[KT, DT, VT, RT], ABC, Generic[KT, DT, VT, RT]):
    @property
    @abstractmethod
    def parent(self) -> Instance[KT, DT, VT, RT]:
        """Retrieve the parent of this instance

        Returns
        -------
        Instance[KT, DT, VT, RT]
            The parent of the instance
        """        
        raise NotImplementedError


class ParentInstance(Instance[KT, DT, VT, RT], ABC, Generic[KT, DT, VT, RT]):
    @property
    @abstractmethod
    def children(self) -> Sequence[ChildInstance[KT, DT, VT, RT]]:
        """Retrieve the children of this instance

        Returns
        -------
        Sequence[ChildInstance[KT, DT, VT, RT]]
            The children instances of this Instance
        """        
        raise NotImplementedError


class InstanceProvider(MutableMapping[KT, Instance[KT, DT, VT, RT]], ABC , Generic[KT, DT, VT, RT]):
    """[summary]

    Parameters
    ----------
    MutableMapping : [type]
        [description]
    ABC : [type]
        [description]
    Generic : [type]
        [description]

    Returns
    -------
    [type]
        [description]

    Yields
    -------
    [type]
        [description]
    """    
    
    @abstractmethod
    def __contains__(self, item: object) -> bool:
        """Special method that checks if something is contained in this 
        provider.

        Parameters
        ----------
        item : object
            The item of which we want to know if it is contained in this
            provider

        Returns
        -------
        bool
            True if the provider contains `item`. 

        Examples
        --------
        Example usage; check if the item exists and then remove it

        >>> doc_id = 20
        >>> provider = InstanceProvider()
        >>> if doc_id in provider:
        ...     del provider[doc_id]
        """        
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[KT]:
        """[summary]

        Yields
        -------
        Iterator[KT]
            [description]

        Raises
        ------
        NotImplementedError
            [description]
        """        
        raise NotImplementedError

    def add(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Add an instance to this provider. If the 
        provider already contains `instance`, nothing happens.

        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be added to the provider
        """        
        self.__setitem__(instance.identifier, instance)

    def discard(self, instance: Instance[KT, DT, VT, RT]) -> None:
        """Remove an instance from this provider. If the 
        provider does not contain `instance`, nothing happens.

        Parameters
        ----------
        instance : Instance[KT, DT, VT, RT]
            The instance that should be removed from the provider
        """        
        try:
            self.__delitem__(instance.identifier)
        except KeyError:
            pass  # To adhere to Set.discard(...) behavior

    @property
    def key_list(self) -> List[KT]:
        """Return a list of all instance keys in this provider

        Returns
        -------
        List[KT]
            A list of instance keys
        """        
        return list(self.keys())

    @property
    @abstractmethod
    def empty(self) -> bool:
        """Determines if the provider does not contain instances

        Returns
        -------
        bool
            True if the provider is empty
        """        
        raise NotImplementedError

    @abstractmethod
    def get_all(self) -> Iterator[Instance[KT, DT, VT, RT]]:
        """Get an iterator that iterates over all instances

        Yields
        ------
        Instance[KT, DT, VT, RT]
            An iterator that iterates over all instances
        """        
        raise NotImplementedError

    @abstractmethod
    def clear(self) -> None:
        """Removes all instances from the provider

        Warning
        -------
        Use this operation with caution! This operation is intended for
        use with providers that function as temporary user queues, not
        for large proportions of the dataset like `unlabeled` and `labeled`
        sets.
        """        
        raise NotImplementedError

    def bulk_add_vectors(self, keys: Sequence[KT], values: Sequence[VT]) -> None:
        """This methods adds vectors in `values` to the instances specified
        in `keys`. 

        In some use cases, vectors are not known beforehand. This library
        provides several :term:`vectorizer` s that convert raw data points
        in feature vector form. Once these vectors are available, they can be 
        added to the provider by using this method

        Parameters
        ----------
        keys : Sequence[KT]
            A sequence of keys
        values : Sequence[VT]
            A sequence of vectors
        
        Warning
        -------
        We assume that the indices and length of the parameters `keys` and `values`
        match.
        """
        for key, vec in zip(keys, values):
            self[key].vector = vec

    def bulk_get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[VT]]:
        """Given a list of instance `keys`, return the vectors

        Parameters
        ----------
        keys : Sequence[KT]
            A list of vectors

        Returns
        -------
        Tuple[Sequence[KT], Sequence[VT]]
            A tuple of two sequences, one with `keys` and one with `vectors`.
            The indices match, so the instance with ``keys[2]`` has as
            vector ``vectors[2]``

        Warning
        -------
        Some underlying implementations do not preserve the ordering of the parameter
        `keys`. Therefore, always use the keys variable from the returned tuple for 
        the correct matching.
        """        
        vector_pairs = ((key, self[key].vector)  for key in keys)
        ret_keys, ret_vectors = filter_snd_none_zipped(vector_pairs)
        return ret_keys, ret_vectors # type: ignore

    def data_chunker(self, batch_size: int) -> Iterator[Sequence[Instance[KT, DT, VT, RT]]]:
        """Iterate over all instances (with or without vectors) in 
        this provider

        Parameters
        ----------
        batch_size : int
            The batch size, the generator will return lists with size `batch_size`

        Yields
        -------
        Sequence[Instance[KT, DT, VT, RT]]]
            A sequence of instances with length `batch_size`. The last list may have
            a shorter length.
        """        
        chunks = divide_iterable_in_lists(self.values(), batch_size)
        yield from chunks

    def vector_chunker(self, batch_size: int) -> Iterator[Sequence[Tuple[KT, VT]]]:
        """Iterate over all pairs of keys and vectors in 
        this provider

        Parameters
        ----------
        batch_size : int
            The batch size, the generator will return lists with size `batch_size`
        
        Returns
        -------
        Iterator[Sequence[Tuple[KT, VT]]]
            An iterator over sequences of key vector tuples
        
        Yields
        -------
        Iterator[Sequence[Tuple[KT, VT]]]
            Sequences of key vector tuples
        """        
        id_vecs = ((elem.identifier, elem.vector) for elem in self.values() if elem.vector is not None)
        chunks = divide_iterable_in_lists(id_vecs, batch_size)
        return chunks

    def bulk_get_all(self) -> List[Instance[KT, DT, VT, RT]]:
        """Returns a list of all instances in this provider.
        
        Returns
        -------
        List[Instance[KT, DT, VT, RT]]
            A list of all instances in this provider

        Warning
        -------
        When using this method on very large providers with lazily loaded instances, this
        may yield Out of Memory errors, as all the data will be loaded into RAM.
        Use with caution!
        """        
        return list(self.get_all())
        