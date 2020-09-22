from typing import Dict

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.multioutput import ClassifierChain
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelBinarizer, MultiLabelBinarizer
from sklearn.svm import LinearSVC

from balancing import BalancerFactory
from balancing.catalog import BalancerType
from machinelearning import MultilabelSkLearnClassifier, SkLearnClassifier

from factory.factory import AbstractBuilder, ObjectFactory
from factory.component import Component
from .catalog import MachineLearningTask, MulticlassMethod, SklearnModel


class ClassifierBuilder(AbstractBuilder):
    def __call__(self, ml_task: MachineLearningTask, **kwargs):
        return self._factory.create(ml_task, **kwargs)

class BinaryClassificationBuilder(AbstractBuilder):
    def __call__(self, 
            sklearn_model: SklearnModel, 
            model_configuration: Dict, 
            balancer_type: BalancerType = BalancerType.IDENTITY, 
            balancer_configuration = {}, 
            storage_location = None, filename=None, **kwargs):
        encoder = LabelBinarizer()
        balancer = self._factory.create(balancer_type, **balancer_configuration)
        classifier = self._factory.create(sklearn_model, **model_configuration)
        return SkLearnClassifier(
            classifier, encoder, balancer = balancer, storage_location=storage_location, filename=filename)

class SklearnBuilder(AbstractBuilder):
    def __call__(self, sk_type: SklearnModel, sklearn_config, **kwargs):
        return self._factory.create(sk_type, **sklearn_config)

class MulticlassBuilder(AbstractBuilder):
    def __call__(self, mc_method: MulticlassMethod, **kwargs):
        return self._factory.create(mc_method, **kwargs)

class MultilabelBuilder(AbstractBuilder):
    def __call__(self, mc_method: MulticlassMethod, **kwargs):
        encoder = MultiLabelBinarizer()
        classifier = self._factory.create(mc_method, **kwargs)
        return MultilabelSkLearnClassifier(classifier, encoder)

class OneVsRestBuilder(AbstractBuilder):
    def __call__(self, sklearn_model: SklearnModel, model_configuration, **kwargs):
        base_classifier = self._factory.create(sklearn_model, **model_configuration)
        return OneVsRestClassifier(base_classifier)

class MachineLearningFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(BalancerFactory())
        self.register_builder(Component.CLASSIFIER, ClassifierBuilder())
        self.register_builder(MachineLearningTask.BINARY, BinaryClassificationBuilder())
        self.register_builder(MachineLearningTask.MULTICLASS, MulticlassBuilder())
        self.register_builder(MachineLearningTask.MULTILABEL, MultilabelBuilder())
        self.register_builder(MulticlassMethod.ONE_VS_REST, OneVsRestBuilder())
        self.register_constructor(SklearnModel.RANDOM_FOREST, RandomForestClassifier)
        self.register_constructor(SklearnModel.NAIVE_BAYES, MultinomialNB)
        self.register_constructor(SklearnModel.LOGISTIC, LogisticRegression)
        self.register_constructor(SklearnModel.SVM, LinearSVC)

