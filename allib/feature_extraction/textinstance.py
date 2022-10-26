from __future__ import annotations

from typing import Sequence, Any

import numpy as np
import numpy.typing as npt

from instancelib import Instance

from .base import BaseVectorizer

InstanceList = Sequence[Instance[Any, str, npt.NDArray[Any], Any]] # type: ignore

class TextInstanceVectorizer(BaseVectorizer[Instance[Any, str, npt.NDArray[Any], Any]]): 
    _name = "TextInstanceVectorizer"
    def __init__(self,
                 vectorizer: BaseVectorizer[str],
                 ) -> None:
        super().__init__()
        self.innermodel = vectorizer

    def fitted(self) -> bool:
        return self.innermodel.fitted

    def fit(self, x_data: InstanceList, **kwargs: Any) -> TextInstanceVectorizer:
        texts = [x.data for x in x_data]
        self.innermodel.fit(texts)
        return self

    def transform(self, x_data: InstanceList, **kwargs: Any) -> npt.NDArray[Any][Any]:
        texts = [x.data for x in x_data]
        return self.innermodel.transform(texts) # type: ignore

    def fit_transform(self, x_data: InstanceList, **kwargs: Any) -> npt.NDArray[Any][Any]:
        texts = [x.data for x in x_data]
        return self.innermodel.fit_transform(texts) # type: ignore
