from typing import Sequence, Generic, TypeVar



from ..instances import Instance

from .base import BaseVectorizer

VT = TypeVar("VT")

class TextInstanceVectorizer(BaseVectorizer, Generic[VT]):
    name = "TextInstanceVectorizer"
    def __init__(self,
                 vectorizer: BaseVectorizer,
                 ) -> None:
        super().__init__()
        self.innermodel = vectorizer

    def fit(self, x_data: Sequence[Instance], **kwargs) -> None:
        texts = [x.data for x in x_data]
        self.innermodel.fit(texts)

    def transform(self, x_data: Sequence[Instance], **kwargs) -> VT:
        texts = [x.data for x in x_data]
        return self.innermodel.transform(texts)

    def fit_transform(self, x_data: Sequence[Instance], **kwargs) -> VT:
        texts = [x.data for x in x_data]
        return self.innermodel.fit_transform(texts)
