from __future__ import annotations

from typing import Generic, Mapping, Sequence, Any, Union

import numpy as np
import numpy.typing as npt

from instancelib import Instance

from abc import abstractmethod
from .base import BaseVectorizer

from typing_extensions import Self
InstanceList = Sequence[Instance[Any, Mapping[str, str], npt.NDArray[Any], Any]] # type: ignore

from instancelib.typehints.typevars import DT
    

class TextVectorizer(BaseVectorizer[Instance[Any, DT, npt.NDArray[Any], Any]], Generic[DT]): 
    _name = "AbstractsVectorizer"
    def __init__(self,
                 vectorizer: BaseVectorizer[str],
                 ) -> None:
        super().__init__()
        self.innermodel = vectorizer

    def adapt_data(self, ins: Instance[Any, DT, Any, Any]) -> str:
        assert isinstance(ins.data, str) 
        return ins.data
    
    def to_str_list(self, inss: Sequence[Instance[Any, DT, Any, Any]]) -> Sequence[str]:
        return [self.adapt_data(x) for x in inss]

    def fitted(self) -> bool:
        return self.innermodel.fitted

    def fit(self, x_data: Sequence[Instance[Any, DT, Any, Any]], **kwargs: Any) -> Self:
        texts = self.to_str_list(x_data)
        self.innermodel.fit(texts)
        return self

    def transform(self, x_data: Sequence[Instance[Any, DT, Any, Any]], **kwargs: Any) -> npt.NDArray[Any]:
        texts = self.to_str_list(x_data)
        return self.innermodel.transform(texts) # type: ignore

    def fit_transform(self, x_data: Sequence[Instance[Any, DT, Any, Any]], **kwargs: Any) -> npt.NDArray[Any]:
        texts = self.to_str_list(x_data)
        return self.innermodel.fit_transform(texts) # type: ignore
    

class AbstractsVectorizer(TextVectorizer[Mapping[str, str]]):
    def adapt_data(self, ins: Instance[Any, Mapping[str, str], Any, Any]) -> str:
        return f'{ins.data["title"]} {ins.data["abstract"]}'
    
class CombinedVectorizer(TextVectorizer[Union[str,Mapping[str, str]]]):
    def adapt_data(self, ins: Instance[Any, Union[str, Mapping[str, str]], Any, Any]) -> str:
        if isinstance(ins.data, str):
            return ins.data
        return f'{ins.data["title"]} {ins.data["abstract"]}'
