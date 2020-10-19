from enum import Enum


class FECatalog:
    class DataType(Enum):
        TEXTINSTANCE = "TextInstance"
        CHATMESSAGES = "ChatMessages"
        EMAIL = "Email"
        IMAGE = "Image"

    class VectorizerType(Enum):
        STACK = "Stack"
        SKLEARN = "Sklearn"
        DOC2VEC = "Doc2Vec"
        DUALVEC = "DualVec"

    class SklearnVecType(Enum):
        TFIDF_VECTORIZER = "TfIdf"
        COUNT_VECTORIZER = "CountVectorizer"
