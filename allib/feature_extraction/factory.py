from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer # type: ignore

from ..factory import AbstractBuilder, ObjectFactory
from ..module.component import Component

from .base import SeparateContextVectorizer, StackVectorizer
from .catalog import FECatalog as FE
#from .contextinstance import ContextVectorizer
from .doc2vec import Doc2VecVectorizer
from .textinstance import TextInstanceVectorizer
from .textsklearn import SklearnVectorizer


class FeatureExtractorBuilder(AbstractBuilder):
    def __call__(self, datatype, **kwargs):
        return self._factory.create(datatype, **kwargs)

class VectorizerBuilder(AbstractBuilder):
    def __call__(self, vec_type, **kwargs):
        return self._factory.create(vec_type, **kwargs)

class TextInstanceVectorizerBuilder(AbstractBuilder):
    def __call__(self, **kwargs):
        vectorizer = self._factory.create(Component.VECTORIZER, **kwargs)
        return TextInstanceVectorizer(vectorizer)

# class ContextInstanceVectorizerBuilder(AbstractBuilder):
#     def __call__(self, **kwargs):
#         vectorizer = self._factory.create(Component.VECTORIZER, **kwargs)
#         return ContextVectorizer(vectorizer)

class StackVectorizerBuilder(AbstractBuilder):
    def __call__(self, vectorizers, **kwargs):
        vectorizers_list = [
            self._factory.create(Component.VECTORIZER, **config)
            for config in vectorizers]
        return StackVectorizer(*vectorizers_list)

class SeparateContextVectorizerBuider(AbstractBuilder):
    def __call__(self, vectorizer_data, vectorizer_context, **kwargs):
        data_vectorizer = self._factory.create(Component.VECTORIZER, **vectorizer_data)
        context_vectorizer = self._factory.create(Component.VECTORIZER, **vectorizer_context)
        return SeparateContextVectorizer(data_vectorizer, context_vectorizer)

class SklearnVectorizerBuilder(AbstractBuilder):
    def __call__(self, sklearn_vec_type, sklearn_config, storage_location = None, **_ignoredkwargs):
        sklearn_vec = self._factory.create(sklearn_vec_type, **sklearn_config)
        return SklearnVectorizer(sklearn_vec, storage_location)

class TfIDFVectorizerBuilder(AbstractBuilder):
    def __call__(self, **configuration) -> TfidfVectorizer:
        return TfidfVectorizer(**configuration)

class CountVectorizerBuilder(AbstractBuilder):
    def __call__(self, **configuration) -> CountVectorizer:
        return CountVectorizer(**configuration)


class FeatureExtractionFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.register_builder(Component.FEATURE_EXTRACTION, FeatureExtractorBuilder())
        self.register_builder(Component.VECTORIZER, VectorizerBuilder())
        # Data extractors
        # self.register_builder(FE.DataType.CHATMESSAGES, 
        #                  ContextInstanceVectorizerBuilder())
        self.register_builder(FE.DataType.TEXTINSTANCE,
                         TextInstanceVectorizerBuilder())
        # Vectorizer Containers
        self.register_builder(FE.VectorizerType.STACK, StackVectorizerBuilder())
        self.register_builder(FE.VectorizerType.SKLEARN, SklearnVectorizerBuilder())
        self.register_builder(FE.VectorizerType.DUALVEC, SeparateContextVectorizerBuider())

        # Actual vectorizers
        self.register_constructor(FE.VectorizerType.DOC2VEC, Doc2VecVectorizer)
        self.register_constructor(FE.SklearnVecType.TFIDF_VECTORIZER, TfidfVectorizer)
        self.register_constructor(FE.SklearnVecType.COUNT_VECTORIZER, CountVectorizer)
