from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.svm import LinearSVC

from ..balancing import BalancerFactory
from ..balancing.catalog import BalancerCatalog as BL
from ..factory import AbstractBuilder, ObjectFactory
from ..machinelearning import (MultilabelSkLearnClassifier,
                                   SkLearnClassifier)
from allib import Component

from .catalog import MachineLearningCatalog as ML


class ClassifierBuilder(AbstractBuilder):
    def __call__(self, task, **kwargs):
        return self._factory.create(task, **kwargs)

class BinaryClassificationBuilder(AbstractBuilder):
    def __call__(self, 
            sklearn_model: ML.SklearnModel, 
            model_configuration: Dict,
            balancer: Dict,  
            storage_location = None, filename=None, **kwargs):
        encoder = LabelBinarizer()
        balancer = self._factory.create(Component.BALANCER, **balancer)
        classifier = self._factory.create(sklearn_model, **model_configuration)
        return SkLearnClassifier(
            classifier, encoder, balancer = balancer, storage_location=storage_location, filename=filename)

class SklearnBuilder(AbstractBuilder):
    def __call__(self, sk_type: ML.SklearnModel, sklearn_config, **kwargs):
        return self._factory.create(sk_type, **sklearn_config)

class MulticlassBuilder(AbstractBuilder):
    def __call__(self, mc_method: ML.MulticlassMethod, **kwargs):
        return self._factory.create(mc_method, **kwargs)

class MultilabelBuilder(AbstractBuilder):
    def __call__(self, mc_method: ML.MulticlassMethod, **kwargs):
        encoder = MultiLabelBinarizer()
        classifier = self._factory.create(mc_method, **kwargs)
        return MultilabelSkLearnClassifier(classifier, encoder)

class OneVsRestBuilder(AbstractBuilder):
    def __call__(self, sklearn_model: ML.SklearnModel, model_configuration, **kwargs):
        base_classifier = self._factory.create(sklearn_model, **model_configuration)
        return OneVsRestClassifier(base_classifier)

class MachineLearningFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(BalancerFactory())
        self.register_builder(Component.CLASSIFIER, ClassifierBuilder())
        self.register_builder(ML.Task.BINARY, BinaryClassificationBuilder())
        self.register_builder(ML.Task.MULTICLASS, MulticlassBuilder())
        self.register_builder(ML.Task.MULTILABEL, MultilabelBuilder())
        self.register_builder(ML.MulticlassMethod.ONE_VS_REST, OneVsRestBuilder())
        self.register_constructor(ML.SklearnModel.RANDOM_FOREST, RandomForestClassifier)
        self.register_constructor(ML.SklearnModel.NAIVE_BAYES, MultinomialNB)
        self.register_constructor(ML.SklearnModel.LOGISTIC, LogisticRegression)
        self.register_constructor(ML.SklearnModel.SVM, LinearSVC)
