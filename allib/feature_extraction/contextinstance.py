# from __future__ import annotations

# from typing import Sequence, Any, Generic, TypeVar

# import numpy as np # type: ignore

# from instancelib.instances.base import ContextInstance
# from ..utils.func import list_unzip

# from .base import BaseVectorizer, SeparateContextVectorizer



# DT = TypeVar("DT")
# CT = TypeVar("CT")
# class ContextVectorizer(BaseVectorizer[ContextInstance[Any, DT, np.ndarray, Any, CT]], Generic[DT, CT]):
#     _name = "ContextVectorizer"
#     def __init__(self,
#                  vectorizer: SeparateContextVectorizer[DT, CT]) -> None:
#         super().__init__()
#         self.innermodel = vectorizer

#     def fitted(self) -> bool:
#         return self.innermodel.fitted

#     def fit(self, 
#             x_data: Sequence[ContextInstance[Any, DT, np.ndarray, Any, CT]], 
#             **kwargs: Any) -> ContextVectorizer[DT, CT]: # type: ignore
#         texts, contexts = list_unzip(((x.data, x.context) for x in x_data))
#         self.innermodel.fit(texts, contexts, **kwargs)
#         return self

#     def transform(self, 
#                   x_data: Sequence[ContextInstance[Any, DT, np.ndarray, Any, CT]], 
#                   **kwargs: Any) -> np.ndarray: # type: ignore
#         texts, contexts = list_unzip(((x.data, x.context) for x in x_data))
#         return self.innermodel.transform(texts, contexts, **kwargs)

#     def fit_transform(self, 
#                       x_data: Sequence[ContextInstance[Any, DT, np.ndarray, Any, CT]], 
#                       **kwargs: Any) -> np.ndarray: # type: ignore
#         self.fit(x_data, **kwargs) # type: ignore
#         return self.transform(x_data, **kwargs) # type: ignore
