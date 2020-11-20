from abc import ABC, abstractmethod
from typing import (Dict, Generic, Iterable, Iterator, MutableMapping,
                    Optional, Sequence, Tuple, TypeVar, Union)

import h5py
import numpy as np  # type: ignore
from h5py._hl.dataset import Dataset

from ..utils.chunks import divide_iterable_in_lists, get_range

KT = TypeVar("KT")
VT = TypeVar("VT")

def slicer(matrix: Union[Dataset, np.ndarray], slices: Iterable[Tuple[int, Optional[int]]]) -> Union[np.ndarray, Dataset]:
        def get_slices_1d():
            for slice_min, slice_max in slices:
                if slice_max is not None:
                    yield matrix[slice_min:slice_max]
                else:
                    yield matrix[slice_min]
        def get_slices_2d():
            for slice_min, slice_max in slices:
                if slice_max is not None:
                    yield matrix[slice_min:slice_max,:]
                else:
                    yield matrix[slice_min,:]
        dims = len(matrix.shape)
        if dims == 1:
            return np.hstack(list(get_slices_1d())) # type: ignore
        return np.vstack(list(get_slices_2d())) # type: ignore

class VectorStorage(MutableMapping[KT, VT], Generic[KT, VT]):
    @abstractmethod
    def __getitem__(self, k: KT) -> VT:
        raise NotImplementedError

    @abstractmethod
    def __setitem__(self, k: KT, value: VT) -> None:
        raise NotImplementedError

    @abstractmethod
    def __contains__(self, item: object) -> bool:
        raise NotImplementedError

    @abstractmethod
    def __iter__(self) -> Iterator[KT]:
        raise NotImplementedError

    @abstractmethod
    def add_bulk(self, keys: Sequence[KT], values: Union[Sequence[VT]]) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_bulk(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], VT]:
        raise NotImplementedError

    @abstractmethod
    def vector_chunker(self) -> Iterator[Tuple[KT, VT]]:
        raise NotImplementedError

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError
    @abstractmethod
    def __exit__(self, type, value, traceback):
        raise NotImplementedError

class H5PYVectorStorage(VectorStorage[KT, np.ndarray], Generic[KT]):
    __writemodes = ["a", "r+", "w", "w-", "x"]
    def __init__(self, h5path: str, mode="r") -> None:
        self.__mode = mode
        self.h5path = h5path
        self.file = h5py.File(self.h5path, self.__mode)
        self.key_dict: Dict[KT, int] = dict()
        self.inv_key_dict: Dict[int, KT] = dict()
        if "keys" in self.file:
            keyset = self.file["keys"]
            assert isinstance(keyset, Dataset)
            keypairs = list(enumerate(keyset))
            self.key_dict = { # type: ignore
                key: i for i, key in keypairs
            }
            self.inv_key_dict = { # type: ignore
                i: key for i, key in keypairs
            }
        
    def __enter__(self):
        return self
    
    def __exit__(self, type, value, traceback):
        self.file.close()

    def _create_matrix(self, first_slice: np.ndarray) -> None:
        assert self.__mode in self.__writemodes
        vector_dim = first_slice.shape[1]
        if "vectors" not in self.file:
            self.file.create_dataset(
                "vectors", data=first_slice, maxshape=(None, vector_dim), dtype="f", chunks=True)

    def _create_keys(self, keys: Sequence[KT]) -> None:
        assert self.__mode in self.__writemodes
        if "keys" not in self.file:
            self.file.create_dataset("keys", 
                data = np.array(keys), maxshape=(None,)) # type: ignore
        for i, key in enumerate(keys):
            self.key_dict[key] = i
            self.inv_key_dict[i] = key
  
    def __len__(self) -> int:
        if "keys" in self.file:
            keyset = self.file["keys"]
            assert isinstance(keyset, Dataset)
            return len(keyset)
        return 0
    
    def _append_matrix(self, matrix: np.ndarray) -> bool:
        assert self.__mode in self.__writemodes
        assert "vectors" in self.file
        dataset = self.file["vectors"]
        assert isinstance(dataset, Dataset)
        old_shape = dataset.shape
        mat_shape = matrix.shape
        assert mat_shape[1] == old_shape[1]
        new_shape = (dataset.shape[0] + mat_shape[0], mat_shape[1])
        dataset.resize(size=new_shape)
        dataset[-mat_shape[0]:,:] = matrix
        self.file.flush()
        return True 

    def _append_keys(self, keys: Sequence[KT]) -> bool:
        assert self.__mode in self.__writemodes
        assert all(map(lambda k: k not in self.key_dict, keys))
        new_keys = np.array(keys) # type: ignore
        key_set = self.file["keys"]
        assert isinstance(key_set, Dataset)
        old_shape = key_set.shape
        arr_shape = new_keys.shape
        new_shape = (old_shape[0] + arr_shape[0],)
        key_set.resize(size=new_shape)
        key_set[-arr_shape[0]:] = new_keys
        start_index = old_shape[0] + 1
        for i, key in enumerate(keys):
            hdf5_idx = start_index + i
            self.key_dict[key] = hdf5_idx
            self.inv_key_dict[hdf5_idx] = key
        self.file.flush()
        return True
        
    def __getitem__(self, k: KT) -> np.ndarray:
        if k in self:
            h5_idx = self.key_dict[k]
            return self.file["vectors"][h5_idx,:] # type: ignore
        raise KeyError 

    def __setitem__(self, k: KT, value: np.ndarray) -> None:
        assert self.__mode in self.__writemodes
        if k in self:
            h5_idx = self.key_dict[k]
            self.file["vectors"][h5_idx] = value # type: ignore
            return
        raise KeyError 

    def __delitem__(self, v: KT) -> None:
        raise NotImplementedError
    
    def __contains__(self, item: object) -> bool:
        return item in self.key_dict
        

    
    def __iter__(self) -> Iterator[KT]:
        yield from self.key_dict

    def add_bulk_matrix(self, keys: Sequence[KT], matrix: np.ndarray) -> None:
        assert self.__mode in self.__writemodes
        assert len(keys) == matrix.shape[0]
        if "vectors" not in self.file and "keys" not in self.file:
            self._create_keys(keys)
            self._create_matrix(matrix)
            return
        if all(map(lambda k: k not in self.key_dict, keys)):
            if self._append_keys(keys):
                self._append_matrix(matrix)
            return

    def _update_vectors(self, keys: Sequence[KT], values: Sequence[np.ndarray]) -> None:
        assert self.__mode in self.__writemodes
        assert len(keys) == len(values)
        for key, value in values:
            self[key] = value
            
    def add_bulk(self, keys: Sequence[KT], values: Sequence[np.ndarray]) -> None:
        assert self.__mode in self.__writemodes
        assert len(keys) == len(values) and len(keys) > 0
        # Check if the vector storage exists
        if "vectors" not in self.file and "keys" not in self.file:
            matrix = np.vstack(values)
            self._create_keys(keys)
            self._create_matrix(matrix)
            return
        
        # Check if the keys do not already exist in storage
        if all(map(lambda k: k not in self.key_dict, keys)):
            # This is the ideal case, all vectors can directly
            # be appended as a matrix
            matrix = np.vstack(values) # type: ignore
            self.add_bulk_matrix(keys, matrix)
            return
        
        # Find out which (key, vector) pairs are already stored
        not_in_storage = filter(lambda kv: kv[0] not in self.key_dict, zip(keys, values))
        in_storage = filter(lambda kv: kv[0] in self.key_dict, zip(keys, values))
        
        # Update the present key vector pairs
        old_keys, updated_vectors = map(list, zip(*in_storage))
        self._update_vectors(old_keys, updated_vectors)

        # Append the new key vector pairs
        new_keys, new_vectors = map(list, zip(*not_in_storage))
        matrix = np.vstack(new_vectors)
        self.add_bulk_matrix(new_keys, matrix)

    def get_bulk(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], np.ndarray]:
        assert "vectors" in self.file and "keys" in self.file
        in_storage = frozenset(self.key_dict).intersection(keys)
        h5py_idxs = map(lambda k: self.key_dict[k], in_storage)
        sorted_keys = sorted(h5py_idxs)
        slices = get_range(sorted_keys)
        dataset = self.file["vectors"]
        assert isinstance(dataset, Dataset)
        result_matrix = slicer(dataset, slices)
        included_keys = list(map(lambda hk: self.inv_key_dict[hk], sorted_keys))
        return included_keys, result_matrix # type: ignore

    def vector_chunker(self, chunksize: int = 200) -> Iterator[Tuple[Sequence[KT], np.ndarray]]:
        assert "vectors" in self.file and "keys" in self.file
        dataset = self.file["vectors"]
        keyset = self.file["keys"]
        assert isinstance(dataset, Dataset) and  isinstance(keyset, Dataset)
        assert len(dataset) == len(keyset)
        hdf5_idxs = range(len(dataset))
        chunks = divide_iterable_in_lists(hdf5_idxs, chunksize)
        for chunk in chunks:
            ranges = get_range(chunk)
            keys: Sequence[KT] = slicer(keyset, ranges).tolist() # type: ignore
            matrix = slicer(dataset, ranges)
            yield keys, matrix # type: ignore
