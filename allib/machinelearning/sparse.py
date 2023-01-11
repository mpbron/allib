# Copyright (C) 2021 The InstanceLib Authors. All Rights Reserved.

# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 3 of the License, or (at your option) any later version.

# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.

# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

import itertools
from pathlib import Path
from typing import (Any, Callable, Dict, Generic, Iterator, List, MutableMapping, Optional, Sequence,
                    Tuple, Union, Iterable)
import h5py  # type: ignore
import numpy as np  # type: ignore
import numpy.typing as npt
from scipy import sparse
from h5py._hl.dataset import Dataset  # type: ignore

from instancelib.exceptions.base import NoVectorsException
from instancelib.utils.chunks import divide_iterable_in_lists, get_range
from instancelib.utils.func import filter_snd_none, list_unzip, identity
from instancelib.utils.numpy import (matrix_to_vector_list, matrix_tuple_to_vectors,
                           matrix_tuple_to_zipped, slicer)
from instancelib.instances.vectorstorage import VectorStorage, ensure_writeable


from instancelib.typehints.typevars import KT, DType

def sparseslicer(matrix: Union[Dataset, npt.NDArray[DType]], slices: Iterable[Tuple[int, Optional[int]]]) -> npt.NDArray[DType]:
    def get_slices_1d(): # type: ignore
        for slice_min, slice_max in slices:
            if slice_max is not None:
                yield matrix[slice_min:slice_max]
            else:
                yield matrix[slice_min]
    def get_slices_2d(): # type: ignore
        for slice_min, slice_max in slices:
            if slice_max is not None:
                yield matrix[slice_min:slice_max,:]
            else:
                yield matrix[slice_min,:]
    dims = len(matrix.shape) #type: ignore
    if dims == 1:
        return sparse.vstack(list(get_slices_1d())) # type: ignore
    return sparse.vstack(list(get_slices_2d())) # type: ignore

class SparseVectorStorage(VectorStorage[KT, npt.NDArray[DType], npt.NDArray[DType]], Generic[KT, DType]):
    """This class provides the handling of spase matrices.
    In many cases, storing dense feature matrices or large sets of vectors
    in memory is not feasible.
    
    This class provides methods that `InstanceProvider` implementations
    can use to ensure that only the vectors needed by some operations are 
    kept in memory. This class enables processing all vector in chunks that
    do fit in memory, enabling ordering all unlabeled instances for very large
    datasets.

    Parameters
    ----------
        path : Path
            The path to the npz  file
        mode : str, optional
            The file mode (see `h5py` documentation), by default "r"
    """
    key_dict: MutableMapping[KT, int]
    inv_key_dict: MutableMapping[int, KT]
        
    matrix: Optional[sparse.csr_matrix]
    klist: List[KT]
    
    def __init__(self) -> None:
        self.klist: List[KT] = list()
        self.key_dict = dict()
        self.inv_key_dict = dict()
        self.matrix = None

    def __len__(self) -> int:
        """Returns the size of the dataset
        Returns
        -------
        int
            The size of the dataset
        """        
        return len(self.key_dict)

    @property
    def datasets_exist(self) -> bool:
        """Check if the HDF5 file contains a dataset

        Returns
        -------
        bool
            True, if the file contains a dataset
        """        
        return self.matrix is not None

    def __enter__(self):
        return self
  
    def __exit__(self, type, value, traceback): # type: ignore
        pass
    
    def close(self) -> None:
        """Close the file and store changes to the index to disk
        """        
        self.__exit__(None, None, None) # type: ignore

    def _create_matrix(self, first_slice: npt.NDArray[DType]) -> None:
        """Create a vectors colum in the HDF5 file and add the
        the vectors in `first_slice`

        Parameters
        ----------
        first_slice : npt.NDArray[DType]
            A matrix
        """        
        self.matrix = sparse.csr_matrix(first_slice)
        
    @ensure_writeable
    def _create_keys(self, keys: Sequence[KT]) -> None:
        """Create a key column in the HDF5 file.

        Parameters
        ----------
        keys : Sequence[KT]
            The keys that should be written
        """
        self.klist = list(keys)
        for i, key in enumerate(keys):
            self.key_dict[key] = i
            self.inv_key_dict[i] = key
  
    def _append_matrix(self, matrix: npt.NDArray[DType]) -> bool:
        """Append a matrix to storage (only for internal use)

        Parameters
        ----------
        matrix : npt.NDArray[DType]
            A matrix. The vector dimension should match with this object

        Returns
        -------
        bool
            [description]

        Raises
        ------
        NoVectorsException
            [description]
        """
        assert self.matrix is not None        
        mat_shape = matrix.shape
        assert mat_shape[1] == self.matrix.shape[1] 
        new_csr = sparse.csr_matrix(matrix)
        self.matrix = sparse.vstack((self.matrix, new_csr)) # type: ignore
        return True 

    @ensure_writeable
    def _append_keys(self, keys: Sequence[KT]) -> bool:
        """Append keys to the vector storage

        Parameters
        ----------
        keys : Sequence[KT]
            The keys that should be appended to storage

        Returns
        -------
        bool
            True, if the operation succeeded

        Raises
        ------
        NoVectorsException
            If there are no vectors in storage, non can be appended
        """        
        assert all(map(lambda k: k not in self.key_dict, keys))
        start_index = len(self.klist)
        for i, key in enumerate(keys):
            idx = start_index + i
            self.key_dict[key] = idx
            self.inv_key_dict[idx] = key
        self.klist += keys
        return True
        
    def __getitem__(self, k: KT) -> npt.NDArray[DType]:
        assert self.matrix is not None
        idx = self.key_dict[k]
        data = self.matrix[idx,:] # type: ignore
        return data # type: ignore

    def __setitem__(self, k: KT, value: npt.NDArray[DType]) -> None:
        assert self.matrix is not None
        if k in self:
            idx = self.key_dict[k]
            self.matrix[idx] = value
            return
        raise KeyError 

    def __delitem__(self, v: KT) -> None:
        raise NotImplementedError
    
    def __contains__(self, item: object) -> bool:
        return item in self.key_dict
        
    def __iter__(self) -> Iterator[KT]:
        yield from self.key_dict

    def add_bulk_matrix(self, keys: Sequence[KT], matrix: npt.NDArray[DType]) -> None:
        """Add matrices in bulk

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifiers. The following should hold: `len(keys) == matrix.shape[0]`
        matrix : npt.NDArray[DType]
            A matrix. The rows should correspond with the identifiers in keys
        """        
        assert len(keys) == matrix.shape[0]
        if not self.datasets_exist:
            self._create_matrix(matrix)
            self._create_keys(keys)
            return
        if all(map(lambda k: k not in self.key_dict, keys)):
            if self._append_keys(keys):
                self._append_matrix(matrix)
            return

    def _update_vectors(self, keys: Sequence[KT], values: Sequence[npt.NDArray[DType]]) -> None:
        """Update vectors in bulk
        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifiers
        values : Sequence[npt.NDArray[DType]]
            A list of new vectors
        """
        assert self.matrix is not None        
        assert len(keys) == len(values)
        if values:
            for key, value in zip(keys, values):
                idx = self.key_dict[key]
                self.matrix[idx] = value # type: ignore
            
    def add_bulk(self, input_keys: Sequence[KT], input_values: Sequence[Optional[npt.NDArray[DType]]]) -> None:
        """Add a bulk of keys and values (vectors) to the vector storage

        Parameters
        ----------
        input_keys : Sequence[KT]
            The keys of the Instances
        input_values : Sequence[Optional[npt.NDArray[DType]]]
            The vectors that correspond with the indices
        """        
        assert len(input_keys) == len(input_values) and len(input_keys) > 0

        # Filter all keys that do not have a vector (input_values may contain None)
        keys, values = filter_snd_none(input_keys, input_values) # type: ignore
        
        if not values:
            return
        
        # Check if the vector storage exists
        if not self.datasets_exist:
            matrix: npt.NDArray[DType] = np.vstack(values) # type: ignore
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
        
        # Update the already present key vector pairs
        old_keys, updated_vectors = list_unzip(in_storage)
        self._update_vectors(old_keys, updated_vectors)

        # Append the new key vector pairs
        new_keys, new_vectors = list_unzip(not_in_storage)
        if new_vectors:
            matrix: npt.NDArray[DType] = np.vstack(new_vectors) # type: ignore
            self.add_bulk_matrix(new_keys, matrix)

    

    def _get_matrix(self, idxs: Sequence[int]) -> Tuple[Sequence[KT], npt.NDArray[DType]]:
        """Return a matrix that correspond with the internal `h5_idxs`.

        Parameters
        ----------
        h5_idxs : Sequence[int]
            A list of internal indices that correspond with the indices

        Returns
        -------
        Tuple[Sequence[KT], npt.NDArray[DType]]
            A tuple containing:

                - The public indices (from the :class:`~allib.instances.InstanceProvider`)
                - A matrix where the rows map to the external indices
        Raises
        ------
        NoVectorsException
            If there are no vectors stored in this object
        """        
        if not self.datasets_exist:
            raise NoVectorsException("There are no vectors stored in this object")
        assert self.matrix is not None
        slices = get_range(idxs)
        result_matrix: npt.NDArray[DType] = sparseslicer(self.matrix, slices) # type: ignore
        included_keys = list(map(lambda idx: self.inv_key_dict[idx], idxs))
        return included_keys, result_matrix # type: ignore

    def get_vectors(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], Sequence[npt.NDArray[DType]]]:
        """Return the vectors that correspond with the `keys` 

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifier keys

        Returns
        -------
        Tuple[Sequence[KT], Sequence[npt.NDArray[DType]]]
            A tuple containing two lists:

                - A list with identifier (order may differ from `keys` argument)
                - A list with vectors 
        """        
        ret_keys, ret_matrix = self.get_matrix(keys)
        ret_vectors = matrix_to_vector_list(ret_matrix)
        return ret_keys, ret_vectors

    def get_matrix(self, keys: Sequence[KT]) -> Tuple[Sequence[KT], npt.NDArray[DType]]:
        """Return a matrix containing the vectors that correspond with the `keys`

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifier keys

        Returns
        -------
        Tuple[Sequence[KT], npt.NDArray[DType]]
            A tuple containing:

                - A list with identifier keys
                    (order may differ from `keys` argument)
                - A matrix containing the vectors 
                    (rows correspond with the returned list)

        Raises
        ------
        NoVectorsException
            If there are no vectors returned
        """        
        if not self.datasets_exist:
            raise NoVectorsException("There are no vectors stored in this object")
        in_storage = frozenset(self.key_dict).intersection(keys)
        idxs = map(lambda k: self.key_dict[k], in_storage)
        sorted_keys = sorted(idxs)
        return self._get_matrix(sorted_keys)

    def get_matrix_chunked(self, 
                           keys: Sequence[KT], 
                           chunk_size: int = 200) -> Iterator[Tuple[Sequence[KT], npt.NDArray[DType]]]:
        """Return matrices in chunks of `chunk_size` containing the vectors requested in `keys`

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifier keys
        chunk_size : int, optional
            The size of the chunks, by default 200

        Yields
        -------
        Tuple[Sequence[KT], npt.NDArray[DType]]
            A tuple containing:

                - A list with identifier keys
                    (order may differ from `keys` argument)
                - A matrix containing the vectors 
                    (rows correspond with the returned list)

        Raises
        ------
        StopIteration
            When there are no more chunks to process
        """        
        if not self.datasets_exist:
            raise StopIteration
        in_storage = frozenset(self.key_dict).intersection(keys)
        h5py_idxs = map(lambda k: self.key_dict[k], in_storage)
        sorted_keys = sorted(h5py_idxs)
        chunks = divide_iterable_in_lists(sorted_keys, chunk_size)
        yield from map(self._get_matrix, chunks)

    def get_vectors_chunked(self, 
                            keys: Sequence[KT], 
                            chunk_size: int = 200
                            ) -> Iterator[Tuple[Sequence[KT], Sequence[npt.NDArray[DType]]]]:
        """Return vectors in chunks of `chunk_size` containing the vectors requested in `keys`

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifier keys
        chunk_size : int, optional
            The size of the chunks, by default 200

        Yields
        -------
        Tuple[Sequence[KT], Sequence[npt.NDArray[DType]]]
            A tuple containing two lists:

                - A list with identifiers (order may differ from `keys` argument)
                - A list with vectors 
        """        
        results = itertools.starmap(matrix_tuple_to_vectors, self.get_matrix_chunked(keys, chunk_size))
        yield from results # type: ignore

    def get_vectors_zipped(self, keys: Sequence[KT], chunk_size: int = 200) -> Iterator[Sequence[Tuple[KT, npt.NDArray[DType]]]]:
        """Return vectors in chunks of `chunk_size` containing the vectors requested in `keys`

        Parameters
        ----------
        keys : Sequence[KT]
            A list of identifier keys
        chunk_size : int, optional
            The size of the chunks, by default 200

        Yields
        -------
        Sequence[Tuple[KT, npt.NDArray[DType]]]
            A list containing tuples of:

                - An identifier (order may differ from `keys` argument)
                - A vector 
        """
        results = itertools.starmap(matrix_tuple_to_zipped, self.get_matrix_chunked(keys, chunk_size))
        yield from results # type: ignore
           

    def vectors_chunker(self, chunk_size: int = 200) -> Iterator[Sequence[Tuple[KT, npt.NDArray[DType]]]]:
        """Return vectors in chunks of `chunk_size`. This generator will yield all vectors contained
        in this object.

        Parameters
        ----------
        chunk_size : int, optional
            The size of the chunks, by default 200


        Yields
        -------
        Sequence[Tuple[KT, npt.NDArray[DType]]]
            A list containing tuples of:

                - An identifier
                - A vector
        """        
        results = itertools.starmap(matrix_tuple_to_zipped, self.matrices_chunker(chunk_size))
        yield from results # type: ignore
           
    def matrices_chunker(self, chunk_size: int = 200):
        """Yield matrices in chunks of `chunk_size` containing all the vectors in this object

        Parameters
        ----------
        chunk_size : int, optional
            The size of the chunks, by default 200

        Yields
        -------
        Tuple[Sequence[KT], npt.NDArray[DType]]
            A tuple containing:

                - A list with identifier keys
                - A matrix containing the vectors 
                    (row indices correspond with the list indices)

        Raises
        ------
        StopIteration
            When there are no more chunks to process
        """        
        if not self.datasets_exist:
            raise StopIteration
        idxs = self.inv_key_dict.keys()
        sorted_keys = sorted(idxs)
        chunks = divide_iterable_in_lists(sorted_keys, chunk_size)
        yield from map(self._get_matrix, chunks)
        
    def reload(self) -> None:
        pass
    
    @property
    def writeable(self) -> bool:
        return True


class NumpyStorage(SparseVectorStorage[KT, DType], Generic[KT, DType]):
    matrix: Optional[npt.NDArray[DType]]
    
    def _create_matrix(self, first_slice: npt.NDArray[DType]) -> None:
        """Create a vectors colum in the HDF5 file and add the
        the vectors in `first_slice`

        Parameters
        ----------
        first_slice : npt.NDArray[DType]
            A matrix
        """        
        self.matrix = first_slice
    
    def _append_matrix(self, matrix: npt.NDArray[DType]) -> bool:
        """Append a matrix to storage (only for internal use)

        Parameters
        ----------
        matrix : npt.NDArray[DType]
            A matrix. The vector dimension should match with this object

        Returns
        -------
        bool
            [description]

        Raises
        ------
        NoVectorsException
            [description]
        """
        assert self.matrix is not None        
        mat_shape = matrix.shape
        assert mat_shape[1] == self.matrix.shape[1] 
        new_csr = matrix
        self.matrix = np.vstack((self.matrix, new_csr)) # type: ignore
        return True
    
    def _get_matrix(self, idxs: Sequence[int]) -> Tuple[Sequence[KT], npt.NDArray[DType]]:
        """Return a matrix that correspond with the internal `h5_idxs`.

        Parameters
        ----------
        h5_idxs : Sequence[int]
            A list of internal indices that correspond with the indices

        Returns
        -------
        Tuple[Sequence[KT], npt.NDArray[DType]]
            A tuple containing:

                - The public indices (from the :class:`~allib.instances.InstanceProvider`)
                - A matrix where the rows map to the external indices
        Raises
        ------
        NoVectorsException
            If there are no vectors stored in this object
        """        
        if not self.datasets_exist:
            raise NoVectorsException("There are no vectors stored in this object")
        assert self.matrix is not None
        slices = get_range(idxs)
        result_matrix: npt.NDArray[DType] = slicer(self.matrix, slices) # type: ignore
        included_keys = list(map(lambda idx: self.inv_key_dict[idx], idxs))
        return included_keys, result_matrix # type: ignore
    

    
    
    