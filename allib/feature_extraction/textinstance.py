from __future__ import annotations

from typing import Sequence, Any

import numpy as np # type: ignore

from ..instances import Instance

from .base import BaseVectorizer

InstanceList = Sequence[Instance[Any, str, np.ndarray, Any]] # type: ignore

class TextInstanceVectorizer(BaseVectorizer[Instance[Any, str, np.ndarray, Any]]): 
    name = "TextInstanceVectorizer"
    def __init__(self,
                 vectorizer: BaseVectorizer[str],
                 ) -> None:
        super().__init__()
        self.innermodel = vectorizer

    def fit(self, x_data: InstanceList, **kwargs: Any) -> TextInstanceVectorizer:
        texts = [x.data for x in x_data]
        self.innermodel.fit(texts)
        return self

    def transform(self, x_data: InstanceList, **kwargs: Any) -> np.ndarray[Any]:
        texts = [x.data for x in x_data]
        return self.innermodel.transform(texts) # type: ignore

    def fit_transform(self, x_data: InstanceList, **kwargs: Any) -> np.ndarray[Any]:
        texts = [x.data for x in x_data]
        return self.innermodel.fit_transform(texts) # type: ignore
