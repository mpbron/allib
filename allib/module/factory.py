from ..activelearning import ActiveLearningFactory
from ..environment import EnvironmentFactory
from ..factory import ObjectFactory
from ..feature_extraction import FeatureExtractionFactory
from .catalog import ModuleCatalog as Cat
from .component import Component


class MainFactory(ObjectFactory):
    def __init__(self) -> None:
        super().__init__()
        self.attach(FeatureExtractionFactory())
        self.attach(ActiveLearningFactory())
        self.attach(EnvironmentFactory())
