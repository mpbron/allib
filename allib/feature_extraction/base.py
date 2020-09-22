from __future__ import annotations

import os

from abc import ABC, abstractmethod
from typing import Generic, Sequence, TypeVar, List, Optional
import functools
import uuid

from sklearn.exceptions import NotFittedError
import numpy as np

DT = TypeVar("DT")
CT = TypeVar("CT")
LT = TypeVar("LT")

class BaseVectorizer(ABC, Generic[DT]):
    fitted: bool
    name = "BaseVectorizer"
    
    def __init__(self):
        self.fitted = False
    
    @abstractmethod
    def fit(self, x_data: Sequence[DT], **kwargs) -> BaseVectorizer:
        pass

    @abstractmethod
    def transform(self, x_data: Sequence[DT], **kwargs) -> np.ndarray:
        pass

    @abstractmethod
    def fit_transform(self, x_data: Sequence[DT], **kwargs) -> np.ndarray:
        pass

class SaveableVectorizer(BaseVectorizer, ABC):
    name = "SaveableVectorizer"

    def __init__(self, storage_location, filename = None):
        super().__init__()
        self.storage_location = storage_location
        self.saved = False
        self.innermodel = None
        if filename is None:
            self.filename = self._generate_random_file_name()
    
    def _generate_random_file_name(self) -> str:
        """Generates a random filename

        Returns
        -------
        str
            A random file name
        """
        gen_uuid = uuid.uuid4()
        filename = f"vectorizer_{self.name}_{gen_uuid}.data"
        return filename

    @property
    def filepath(self) -> Optional[str]:
        if self.storage_location is not None:
            return os.path.join(self.storage_location, self.filename)
        return None

    @property
    def has_storage_location(self) -> bool:
        return self.storage_location is not None

    @property
    def is_stored(self) -> bool:
        return self.saved
  
    @staticmethod
    def load_model_fallback(func):
        @functools.wraps(func)
        def wrapper(self: SaveableVectorizer, *args, **kwargs):
            if not self.is_loaded and self.is_stored:
                self.load()
            return func(self, *args, **kwargs)
        return wrapper

    @property
    def is_loaded(self) -> bool:
        return self.innermodel is not None
    
    @abstractmethod
    def save(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    def __getstate__(self):
        if not self.has_storage_location:
            return self.__dict__
        self.save()
        state = {key: value for (key, value) in self.__dict__.items() if key != "innermodel"}
        state = {**state.copy(), **{"innermodel": None}}
        return state
    
class SeparateContextVectorizer(ABC, Generic[DT, CT]):
    fitted: bool
    name = "SeparateContextVectorizer"
    
    def __init__(
            self,
            data_vectorizer: BaseVectorizer,
            context_vectorizer: BaseVectorizer,
            **kwargs):
        self.fitted = False
        self.data_vectorizer = data_vectorizer
        self.context_vectorizer = context_vectorizer

    def fit(
            self,
            x_data: Sequence[DT],
            context_data: Sequence[CT],
            **kwargs):
        self.data_vectorizer.fit(x_data, **kwargs)
        self.context_vectorizer.fit(context_data, **kwargs)
        self.fitted = True

    def transform(
            self,
            x_data: Sequence[DT],
            context_data: Sequence[CT],
            **kwargs) -> np.ndarray:
        if self.fitted:
            data_part = self.data_vectorizer.transform(x_data, **kwargs)
            context_part = self.context_vectorizer.transform(
                context_data, **kwargs)
            return np.concatenate((data_part, context_part), axis=1)
        raise NotFittedError

    def fit_transform(
            self,
            x_data: Sequence[DT],
            context_data: Sequence[CT],
            **kwargs) -> np.ndarray:
        self.fit(x_data, **kwargs)
        return self.transform(context_data, **kwargs)


class StackVectorizer(BaseVectorizer, Generic[DT]):
    vectorizers: List[BaseVectorizer]
    name = "StackVectorizer"

    def __init__(self,
                 vectorizer: BaseVectorizer,
                 *vectorizers: BaseVectorizer) -> None:
        super().__init__()
        self.vectorizers = [vectorizer, *vectorizers]

    def fit(self, x_data: Sequence[DT], **kwargs) -> None:
        for vec in self.vectorizers:
            vec.fit(x_data, **kwargs)
        self.fitted = True

    def transform(self, x_data: Sequence[DT], **kwargs) -> np.ndarray:
        if self.fitted:
            sub_vectors = [
                vec.transform(x_data, **kwargs) 
                for vec in self.vectorizers]
            return np.concatenate(sub_vectors, axis=1)
        raise NotFittedError

    def fit_transform(self, x_data: Sequence[DT], **kwargs) -> np.ndarray:
        self.fit(x_data, **kwargs)
        return self.transform(x_data, **kwargs)
