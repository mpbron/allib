from abc import abstractmethod
from typing import (Dict, Generic, Iterable, Iterator, MutableMapping,
                    Optional, Sequence, Tuple, TypeVar, Union)

import h5py
import pickle
import numpy as np  # type: ignore
from h5py._hl.dataset import Dataset

from ..utils.chunks import divide_iterable_in_lists, get_range

KT = TypeVar("KT")
VT = TypeVar("VT")

def slicer(matrix: Union[Dataset, np.ndarray], slices: Iterable[Tuple[int, Optional[int]]]) -> np.ndarray:
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
    def get_matrix(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], VT]:
        raise NotImplementedError

    @abstractmethod
    def matrices_chunker(self) -> Iterator[Tuple[KT, VT]]:
        raise NotImplementedError

    @abstractmethod
    def __enter__(self):
        raise NotImplementedError
    @abstractmethod
    def __exit__(self, type, value, traceback):
        raise NotImplementedError

class HDF5VectorStorage(VectorStorage[KT, np.ndarray], Generic[KT]):
    __writemodes = ["a", "r+", "w", "w-", "x"]
    def __init__(self, h5path: str, mode="r") -> None:
        self.__mode = mode
        self.h5path = h5path
        self.key_dict: Dict[KT, int] = dict()
        self.inv_key_dict: Dict[int, KT] = dict()
        with h5py.File(self.h5path, self.__mode) as hfile:
            if "dicts" in hfile:
                dicts = hfile["dicts"]
                assert isinstance(dicts, Dataset)
                self.key_dict = pickle.loads(dicts[0]) # type: ignore
                self.inv_key_dict = pickle.loads(dicts[1]) # type: ignore
            
    def __enter__(self):
        return self

    def __store_dicts(self) -> None:
        assert self.__mode in self.__writemodes
        with h5py.File(self.h5path, self.__mode) as hfile:
            if "dicts" not in hfile:
                dt = h5py.special_dtype(vlen=np.dtype("uint8"))
                hfile.create_dataset("dicts", (2,), dtype=dt)
            dicts = hfile["dicts"]
            assert isinstance(dicts, Dataset)
            dicts[0] = np.fromstring(
                pickle.dumps(self.key_dict), dtype="uint8") #type: ignore
            dicts[1] = np.fromstring(
                pickle.dumps(self.inv_key_dict), dtype="uint8") # type: ignore
            
    def __exit__(self, type, value, traceback):
        if self.__mode in self.__writemodes:
            self.__store_dicts()

    @property
    def datasets_exist(self) -> bool:
        with h5py.File(self.h5path, self.__mode) as hfile:
            exist = "vectors" in hfile and "keys" in hfile
        return exist

    def close(self) -> None:
        self.__exit__(None, None, None)

    def _create_matrix(self, first_slice: np.ndarray) -> None:
        assert self.__mode in self.__writemodes
        vector_dim = first_slice.shape[1]
        with h5py.File(self.h5path, self.__mode) as hfile:
            if "vectors" not in hfile:
                hfile.create_dataset(
                    "vectors", data=first_slice, 
                    maxshape=(None, vector_dim), dtype="f", chunks=True)

    def _create_keys(self, keys: Sequence[KT]) -> None:
        assert self.__mode in self.__writemodes
        with h5py.File(self.h5path, self.__mode) as hfile:
            if "keys" not in hfile:
                hfile.create_dataset("keys", 
                    data = np.array(keys), maxshape=(None,)) # type: ignore
            for i, key in enumerate(keys):
                self.key_dict[key] = i
                self.inv_key_dict[i] = key
  
    def __len__(self) -> int:
        return len(self.key_dict)
    
    def _append_matrix(self, matrix: np.ndarray) -> bool:
        assert self.__mode in self.__writemodes
        assert self.datasets_exist
        with h5py.File(self.h5path, self.__mode) as hfile:
            dataset = hfile["vectors"]
            assert isinstance(dataset, Dataset)
            old_shape = dataset.shape
            mat_shape = matrix.shape
            assert mat_shape[1] == old_shape[1]
            new_shape = (dataset.shape[0] + mat_shape[0], mat_shape[1])
            dataset.resize(size=new_shape)
            dataset[-mat_shape[0]:,:] = matrix
        return True 

    def _append_keys(self, keys: Sequence[KT]) -> bool:
        assert self.__mode in self.__writemodes
        assert self.datasets_exist
        assert all(map(lambda k: k not in self.key_dict, keys))
        new_keys = np.array(keys) # type: ignore
        with h5py.File(self.h5path, self.__mode) as hfile:
            key_set = hfile["keys"]
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
        self.__store_dicts()
        return True
        
    def __getitem__(self, k: KT) -> np.ndarray:
        assert self.datasets_exist
        h5_idx = self.key_dict[k]
        with h5py.File(self.h5path, self.__mode) as hfile:
            dataset = hfile["vectors"]
            assert isinstance(dataset, Dataset)
            data = dataset[h5_idx,:]
        return data # type: ignore

    def __setitem__(self, k: KT, value: np.ndarray) -> None:
        assert self.__mode in self.__writemodes
        assert self.datasets_exist
        if k in self:
            h5_idx = self.key_dict[k]
            with h5py.File(self.h5path, self.__mode) as hfile:
                dataset = hfile["vectors"]
                assert isinstance(dataset, Dataset)
                dataset[h5_idx] = value # type: ignore
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
        if not self.datasets_exist:
            self._create_matrix(matrix)
            self._create_keys(keys)
            return
        if all(map(lambda k: k not in self.key_dict, keys)):
            if self._append_keys(keys):
                self._append_matrix(matrix)
            return

    def _update_vectors(self, keys: Sequence[KT], values: Sequence[np.ndarray]) -> None:
        assert self.__mode in self.__writemodes
        assert len(keys) == len(values)
        with h5py.File(self.h5path, self.__mode) as hfile:
            dataset = hfile["vectors"]
            assert isinstance(dataset, Dataset)
            for key, value in values:
                h5_idx = self.key_dict[key]
                dataset[h5_idx] = value # type: ignore
            
    def add_bulk(self, keys: Sequence[KT], values: Sequence[np.ndarray]) -> None:
        assert self.__mode in self.__writemodes
        assert len(keys) == len(values) and len(keys) > 0
        # Check if the vector storage exists
        if not self.datasets_exist:
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

    def _get_matrix(self, h5_idxs: Sequence[int]) -> Tuple[Sequence[KT], np.ndarray]:
        with h5py.File(self.h5path, self.__mode) as dfile:
            dataset = dfile["vectors"]
            assert isinstance(dataset, Dataset)
            slices = get_range(h5_idxs)
            result_matrix = slicer(dataset, slices)
            included_keys = list(map(lambda idx: self.inv_key_dict[idx], h5_idxs))
        return included_keys, result_matrix

    def get_matrix(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], np.ndarray]:
        assert self.datasets_exist
        in_storage = frozenset(self.key_dict).intersection(keys)
        h5py_idxs = map(lambda k: self.key_dict[k], in_storage)
        sorted_keys = sorted(h5py_idxs)
        return self._get_matrix(sorted_keys)

    def get_matrix_chunked(self, keys: Sequence[KT], chunk_size: int = 200) -> Iterator[Tuple[Sequence[KT], np.ndarray]]:
        assert self.datasets_exist
        in_storage = frozenset(self.key_dict).intersection(keys)
        h5py_idxs = map(lambda k: self.key_dict[k], in_storage)
        sorted_keys = sorted(h5py_idxs)
        chunks = divide_iterable_in_lists(sorted_keys, chunk_size)
        yield from map(self._get_matrix, chunks)
           
    def matrices_chunker(self, chunk_size: int = 200) -> Iterator[Tuple[Sequence[KT], np.ndarray]]:
        assert self.datasets_exist
        h5py_idxs = self.inv_key_dict.keys()
        sorted_keys = sorted(h5py_idxs)
        chunks = divide_iterable_in_lists(sorted_keys, chunk_size)
        yield from map(self._get_matrix, chunks)
