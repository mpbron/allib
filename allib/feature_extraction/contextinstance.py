from __future__ import annotations

from typing import Sequence, Any, Generic, TypeVar

import numpy as np # type: ignore

from ..instances.base import ContextInstance
from .base import BaseVectorizer, SeparateContextVectorizer

DT = TypeVar("DT")

InstanceList = Sequence[ContextInstance[Any, DT, np.ndarray, Any, DT]] # type: ignore

class ContextVectorizer(BaseVectorizer[ContextInstance[Any, DT, np.ndarray, Any, DT]], Generic[DT]):
    name = "ContextVectorizer"
    def __init__(self,
                 vectorizer: SeparateContextVectorizer[DT, DT]) -> None:
        super().__init__()
        self.innermodel = vectorizer

    def fit(self, x_data: InstanceList, **kwargs: Any) -> ContextVectorizer[DT]: # type: ignore
        texts, contexts = zip(*((x.data, x.context) for x in x_data)) # type: ignore
        self.innermodel.fit(texts, contexts)
        return self

    def transform(self, x_data: InstanceList, **kwargs: Any) -> np.ndarray: # type: ignore
        tuples = ((x.data, x.context) for x in x_data) # type : ignore
        texts, contexts = zip(*tuples)
        return self.innermodel.transform(texts, contexts, **kwargs) # type: ignore

    def fit_transform(self, x_data: InstanceList, **kwargs: Any) -> np.ndarray: # type: ignore
        self.fit(x_data, **kwargs) # type: ignore
        return self.transform(x_data, **kwargs) # type: ignore
