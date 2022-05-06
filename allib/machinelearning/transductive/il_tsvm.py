from os import PathLike
from typing import Generic, Iterable, Optional

from instancelib.labels.encoder import DictionaryEncoder
import instancelib as il
import numpy as np

from allib.typehints.typevars import IT, KT, LT
from .tsvm_sklearn import SKTSVM

class TSVMEncoder(DictionaryEncoder[LT]):
    def encode(self, labels: Iterable[LT]) -> np.ndarray:
        label_list = list(labels)
        if not label_list:
            return np.array([-1])
        return super().encode(label_list)

    def initialize(self, labels: Iterable[LT]) -> None:
        return super().initialize(labels)

    
class TSVM(il.SkLearnVectorClassifier[IT, KT, LT], Generic[IT, KT, LT]):
    def __init__(self, 
                 storage_location: "Optional[PathLike[str]]" = None, 
                 filename: "Optional[PathLike[str]]" = None) -> None:
        estimator = SKTSVM()
        encoder = TSVMEncoder.from_list([])
        super().__init__(estimator, encoder, storage_location, filename)

    def _filter_x_only_encoded_y(self, instances: Iterable[_T], labelings: Sequence[Iterable[LT]]) -> Tuple[Iterable[_T], np.ndarray]:
        return super()._filter_x_only_encoded_y(instances, labelings)