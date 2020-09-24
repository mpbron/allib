from __future__ import annotations

import pickle

from typing import Sequence

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.exceptions import NotFittedError

from ..utils import SaveableInnerModel
from .base import BaseVectorizer


class SklearnVectorizer(BaseVectorizer, SaveableInnerModel):
    innermodel: BaseEstimator
    name = "SklearnVectorizer"

    def __init__(
        self,
        vectorizer: BaseEstimator,
        storage_location = None,
        filename = None
        ) -> None:
        BaseVectorizer.__init__(self)
        SaveableInnerModel.__init__(self, vectorizer, storage_location, filename)

    @SaveableInnerModel.load_model_fallback
    def fit(self, x_data: Sequence[str], **kwargs) -> SklearnVectorizer:
        self.innermodel = self.innermodel.fit(x_data)
        self.fitted = True
        return self

    @SaveableInnerModel.load_model_fallback
    def transform(self, x_data: Sequence[str], **kwargs) -> np.ndarray:
        if self.fitted:
            # TODO Check for performance issues with .toarray()
            return self.innermodel.transform(x_data).toarray()
        raise NotFittedError

    def fit_transform(self, x_data: Sequence[str], **kwargs) -> np.ndarray:
        self.fit(x_data)
        return self.transform(x_data)
    