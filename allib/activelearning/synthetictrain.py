import itertools
from typing import FrozenSet, Generic, Iterable

from instancelib.machinelearning.base import AbstractClassifier

from ..utils.func import union

from ..environment.base import AbstractEnvironment

from ..typehints.typevars import IT
from instancelib.pertubations.base import ProviderPertubator
from instancelib.typehints.typevars import DT, KT, LT, RT, VT, LMT, PMT


class SyntheticTrainMixIn(Generic[IT, KT, DT, VT, RT, LT, LMT, PMT]):
    
    env: AbstractEnvironment[IT, KT, DT, VT, RT, LT]
    synthesizer: ProviderPertubator[IT, KT, DT, VT, RT]
    synthetic_labels: FrozenSet[LT]
    classifier: AbstractClassifier[IT, KT, DT, VT, RT, LT, LMT, PMT]
    
    def __init__(self, synthesizer: ProviderPertubator[IT, KT, DT, VT, RT], synthetic_labels: Iterable[LT]) -> None:
        self.synthesizer = synthesizer
        self.synthetic_labels = frozenset(synthetic_labels)
    
    def retrain(self) -> None:
        """Retrain the classifier using the information in the labeled set
        and generated instances by the synthesizer

        Raises
        ------
        NotInitializedException
            If the AL method has no attached Environment
        """
        minority_ids = union(*(self.env.labels.get_instances_by_label(label) for label in self.synthetic_labels))
        minority_labeled = minority_ids.intersection(self.env.labeled)
        minority_provider = self.env.create_bucket(minority_labeled)
        synthetic_data = self.synthesizer(minority_provider)
        train_provider = self.env.combine(self.env.labeled, synthetic_data)
        self.classifier.fit_provider(train_provider, self.env.labels)

