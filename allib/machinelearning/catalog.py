from enum import Enum

class MachineLearningCatalog:
    class Task(Enum):
        SEQUENCE = "Sequence"
        BINARY = "Binary"
        MULTICLASS = "MultiClass"
        MULTILABEL = "MultiLabel"

    class MulticlassMethod(Enum):
        ONE_VS_REST = "OneVsRest"
        BINARY_RELEVANCE = "BinaryRelevance"

    class SklearnModel(Enum):
        RANDOM_FOREST = "RandomForest"
        NAIVE_BAYES = "NaiveBayes"
        SVM = "SVM"
        SVC = "SVC"
        LOGISTIC = "LogisticRegression"
        LGBM = "LGBMClassifier"