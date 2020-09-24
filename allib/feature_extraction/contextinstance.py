from typing import Sequence

import numpy as np

from ..instances.base import ContextInstance
from .base import BaseVectorizer, SeparateContextVectorizer

class ContextVectorizer(BaseVectorizer):
    name = "ContextVectorizer"
    def __init__(self,
                 vectorizer: SeparateContextVectorizer) -> None:
        super().__init__()
        self.innermodel = vectorizer

    def fit(self, x_data: Sequence[ContextInstance], **kwargs) -> None:
        texts, contexts = zip(*((x.data, x.context) for x in x_data))
        self.innermodel.fit(texts, contexts)

    def transform(self, x_data: Sequence[ContextInstance], **kwargs) -> np.ndarray:
        tuples = ((x.data, x.context) for x in x_data)
        texts, contexts = zip(*tuples)
        return self.innermodel.transform(texts, contexts, **kwargs)

    def fit_transform(self, x_data: Sequence[ContextInstance], **kwargs) -> np.ndarray:
        self.fit(x_data, **kwargs)
        return self.transform(x_data, **kwargs)
