from __future__ import annotations

from typing import Sequence, Optional, Callable, List, Dict, Any
from tempfile import NamedTemporaryFile

import numpy as np # type: ignore
from sklearn.exceptions import NotFittedError # type: ignore

from gensim.models.doc2vec import Doc2Vec, TaggedLineDocument # type: ignore
from gensim.utils import save_as_line_sentence # type: ignore

from .base import BaseVectorizer
from ..utils import SaveableInnerModel


def get_line_docs(documents: Sequence[str]) -> TaggedLineDocument:
    file = NamedTemporaryFile(delete=False, mode="w+b")
    file.close()
    corpus = [[element] for element in documents]
    save_as_line_sentence(corpus, file.name)
    tld = TaggedLineDocument(file.name)
    return tld


def split_tokenizer(text: str) -> List[str]:
    return text.split(' ')


DocTokenizer = Callable[..., List[str]]


class Doc2VecVectorizer(BaseVectorizer[str], SaveableInnerModel[Doc2Vec]):
    _name = "Doc2Vec"
    innermodel: Optional[Doc2Vec]
    tokenizer: DocTokenizer

    def __init__(
        self,
        d2v_params: Dict[str, Any],
        tokenizer: DocTokenizer = split_tokenizer,
        storage_location: Optional[str]=None,
        filename: Optional[str] = None
    ) -> None:
        BaseVectorizer.__init__(self) # type: ignore
        self.tokenizer = tokenizer # type: ignore
        self.d2v_params = d2v_params
        self.innermodel = None
        SaveableInnerModel.__init__(self, self.innermodel, storage_location, filename)

    def fit(self, x_data: Sequence[str], **kwargs: Any) -> Doc2VecVectorizer:
        self.innermodel = Doc2Vec(
            documents=get_line_docs(x_data), **self.d2v_params)
        self.innermodel.delete_temporary_training_data( # type: ignore
            keep_doctags_vectors=True, keep_inference=True) 
        self._fitted = True
        return self

    @SaveableInnerModel.load_model_fallback
    def transform(self, x_data: Sequence[str], **kwargs: Any) -> np.ndarray: # type: ignore
        if self.fitted and self.innermodel is not None:
            return np.array( # type: ignore
                [
                    self.innermodel.infer_vector(self.tokenizer(doc)) # type: ignore
                    for doc in x_data
                ]) 
        raise NotFittedError

    def fit_transform(self, x_data: Sequence[str], **kwargs: Any) -> np.ndarray: # type: ignore
        self.fit(x_data)
        return self.transform(x_data) # type: ignore

    def save(self) -> None:
        if self.innermodel is not None:
            self.innermodel.save(self.filepath) # type: ignore
            self.saved = True

    def load(self) -> None:
        self.innermodel = Doc2Vec.load(self.filepath) # type: ignore