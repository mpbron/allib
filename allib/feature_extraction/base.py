from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar, List, Any

from sklearn.exceptions import NotFittedError # type: ignore
import numpy as np # type: ignore

DT = TypeVar("DT")
CT = TypeVar("CT")
LT = TypeVar("LT")

class BaseVectorizer(ABC, Generic[DT]):
    fitted: bool
    name = "BaseVectorizer"
    
    def __init__(self):
        self.fitted = False
    
    @abstractmethod
    def fit(self, x_data: Sequence[DT], **kwargs: Any) -> BaseVectorizer[DT]:
        pass

    @abstractmethod
    def transform(self, x_data: Sequence[DT], **kwargs: Any) -> np.ndarray: # type: ignore
        pass

    @abstractmethod
    def fit_transform(self, x_data: Sequence[DT], **kwargs: Any) -> np.ndarray: # type: ignore
        pass

class SeparateContextVectorizer(ABC, Generic[DT, CT]):
    fitted: bool
    name = "SeparateContextVectorizer"
    
    def __init__(
            self,
            data_vectorizer: BaseVectorizer[DT],
            context_vectorizer: BaseVectorizer[CT]
        ):
        self.fitted = False
        self.data_vectorizer = data_vectorizer
        self.context_vectorizer = context_vectorizer

    def fit(
            self,
            x_data: Sequence[DT],
            context_data: Sequence[CT],
            **kwargs: Any) -> SeparateContextVectorizer[DT, CT]:
        self.data_vectorizer.fit(x_data, **kwargs)
        self.context_vectorizer.fit(context_data, **kwargs)
        self.fitted = True
        return self

    def transform(
            self,
            x_data: Sequence[DT],
            context_data: Sequence[CT],
            **kwargs: Any) -> np.ndarray: # type: ignore
        if self.fitted:
            data_part: np.ndarray = self.data_vectorizer.transform(x_data, **kwargs) # type: ignore
            context_part: np.ndarray = self.context_vectorizer.transform( # type: ignore
                context_data, **kwargs) # type: ignore
            return np.concatenate((data_part, context_part), axis=1) # type: ignore
        raise NotFittedError

    def fit_transform(
            self,
            x_data: Sequence[DT],
            context_data: Sequence[CT],
            **kwargs: Any) -> np.ndarray: # type: ignore
        self.fit(x_data, **kwargs)
        return self.transform(x_data, context_data, **kwargs) # type: ignore


class StackVectorizer(BaseVectorizer[DT], Generic[DT]):
    vectorizers: List[BaseVectorizer[DT]]
    name = "StackVectorizer"

    def __init__(self,
                 vectorizer: BaseVectorizer[DT],
                 *vectorizers: BaseVectorizer[DT]) -> None:
        super().__init__()
        self.vectorizers = [vectorizer, *vectorizers]

    def fit(self, x_data: Sequence[DT], **kwargs: Any) -> StackVectorizer[DT]:
        for vec in self.vectorizers:
            vec.fit(x_data, **kwargs)
        self.fitted = True
        return self

    def transform(self, x_data: Sequence[DT], **kwargs: Any) -> np.ndarray: # type: ignore
        if self.fitted:
            sub_vectors = [ # type: ignore
                vec.transform(x_data, **kwargs) # type: ignore
                for vec in self.vectorizers]
            return np.concatenate(sub_vectors, axis=1) # type: ignore
        raise NotFittedError

    def fit_transform(self, x_data: Sequence[DT], **kwargs: Any) -> np.ndarray: # type: ignore
        self.fit(x_data, **kwargs)
        return self.transform(x_data, **kwargs) # type: ignore
