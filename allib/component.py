from enum import Enum

class Component(str, Enum):
    FEATURE_EXTRACTION = "FeatureExtraction"
    CLASSIFIER = "Classifier"
    ACTIVELEARNER = "ActiveLearner"
    ENVIRONMENT = "Environment"
    VECTORIZER = "Vectorizer"
    BALANCER = "Balancer"