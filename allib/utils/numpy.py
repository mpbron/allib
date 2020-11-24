from typing import Iterable, Optional, Sequence, Tuple, TypeVar, Union

from h5py._hl.dataset import Dataset

import numpy as np  # type: ignore

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

def matrix_to_vector_list(matrix: np.ndarray) -> Sequence[np.ndarray]:
    def get_vector(index: int) -> np.ndarray:
        return matrix[index, :]
    n_rows = matrix.shape[0]
    rows = range(n_rows)
    return list(map(get_vector, rows))

def matrix_tuple_to_vectors(keys: Sequence[KT], 
                            matrix: np.ndarray
                           ) -> Tuple[Sequence[KT], Sequence[np.ndarray]]:
    return keys, matrix_to_vector_list(matrix)

def matrix_tuple_to_zipped(keys: Sequence[KT], 
                           matrix: np.ndarray) -> Sequence[Tuple[KT, np.ndarray]]:
    result = list(zip(keys, matrix_to_vector_list(matrix)))
    return result
