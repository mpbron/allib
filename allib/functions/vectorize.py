import itertools
from typing import TypeVar, Sequence, List

import numpy as np  # type: ignore

from ..environment import AbstractEnvironment
from ..feature_extraction import BaseVectorizer
from ..instances import Instance
from ..utils import divide_sequence
from ..utils.to_key import to_key
from ..utils.numpy import matrix_tuple_to_vectors

KT = TypeVar("KT")
DT = TypeVar("DT")
LT = TypeVar("LT")
RT = TypeVar("RT")
VT = TypeVar("VT")

def vectorize(vectorizer: BaseVectorizer[Instance[KT, DT, np.ndarray, RT]], 
              environment: AbstractEnvironment[KT, DT, np.ndarray, RT,  LT], chunk_size: int) -> None:
    instances = list(itertools.chain.from_iterable(environment.dataset.data_chunker(chunk_size)))
    vectorizer.fit(instances)
    def set_vectors(instances: Sequence[Instance[KT, DT, np.ndarray, RT]], batch_size = chunk_size) -> None:
        instance_chunks = divide_sequence(instances, batch_size)
        for instance_chunk in instance_chunks:
            matrix = vectorizer.transform(instance_chunk)
            keys: List[KT] = list(map(to_key, instance_chunk))
            ret_keys, vectors = matrix_tuple_to_vectors(keys, matrix)
            environment.add_vectors(ret_keys, vectors)
    set_vectors(instances)
