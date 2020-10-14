from aenum import Enum

class Component(Enum):
    FEATURE_EXTRACTION = "FeatureExtraction"
    CLASSIFIER = "Classifier"
    ACTIVELEARNER = "ActiveLearner"
    ENVIRONMENT = "Environment"
    VECTORIZER = "Vectorizer"
    BALANCER = "Balancer"