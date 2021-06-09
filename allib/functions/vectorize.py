import itertools
from typing import TypeVar, Sequence, List, Any

import numpy as np  # type: ignore

from ..environment import AbstractEnvironment
from ..feature_extraction import BaseVectorizer
from instancelib.instances import Instance
from ..utils import divide_sequence
from ..utils.to_key import to_key
from ..utils.numpy import matrix_tuple_to_vectors

def vectorize(vectorizer: BaseVectorizer[Instance[Any, Any, np.ndarray, Any]], 
              environment: AbstractEnvironment[Instance[Any, Any, np.ndarray, Any], Any, Any, np.ndarray, Any, Any],
              fit: bool = True, 
              chunk_size: int = 200) -> None:
    def fit_vector() -> None:
        instances = list(
            itertools.chain.from_iterable(
                environment.all_datapoints.data_chunker(chunk_size)))
        vectorizer.fit(instances)
    def set_vectors() -> None:
        instance_chunks = environment.all_datapoints.data_chunker(chunk_size)
        for instance_chunk in instance_chunks:
            matrix = vectorizer.transform(instance_chunk)
            keys: List[KT] = list(map(to_key, instance_chunk)) # type: ignore
            ret_keys, vectors = matrix_tuple_to_vectors(keys, matrix)
            environment.add_vectors(ret_keys, vectors)
    if fit:
        fit_vector()
    set_vectors()
