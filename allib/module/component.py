from enum import Enum

class Component(Enum):
    FEATURE_EXTRACTION = "FeatureExtraction"
    CLASSIFIER = "Classifier"
    ACTIVELEARNER = "ActiveLearner"
    FALLBACK = "Fallback"
    ENVIRONMENT = "Environment"
    VECTORIZER = "Vectorizer"
    BALANCER = "Balancer"
    SELECTION_CRITERION = "SelectionCriterion"