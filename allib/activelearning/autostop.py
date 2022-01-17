from typing import Generic, Optional, Sequence, Tuple

import instancelib as il
import numpy as np
from instancelib.typehints import DT, KT, LT, RT

from allib.activelearning.random import RandomSampling
import random

from allib.utils.numpy import raw_proba_chainer

from ..typehints import IT
from ..typehints.typevars import VT
from .insclass import ILLabelProbabilityBased, ILMLBased, ILProbabilityBased
from .poolbased import PoolBasedAL
from .selectioncriterion import AbstractSelectionCriterion


class AutoStopLearner(PoolBasedAL[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, RT, LT]):
    
    def __init__(self, 
                 classifier: il.AbstractClassifier[IT, KT, DT, VT, RT, LT, np.ndarray, np.ndarray],
                 pos_label: LT, 
                 neg_label: LT, 
                 k_sample: int,
                 batch_size: int,  
                 *_, identifier: Optional[str] = None, **__) -> None:
        super().__init__(*_, identifier=identifier, **__)
        self.classifier = classifier
        self.batch_it = 0
        self.trained: bool = False
        self.k_sample = k_sample
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.batch_size = batch_size
    
    def update_ordering(self) -> bool:
        return True

    def _provider_sample(self, provider: il.InstanceProvider[IT, KT, DT, np.ndarray, RT]) -> il.InstanceProvider[IT, KT, DT, np.ndarray, RT]:
        sampled_keys = random.sample(provider.key_list, self.k_sample)
        sampled_provider = self.env.create_bucket(sampled_keys)
        return sampled_provider

    def _temp_augment(self):
        temp_labels = il.MemoryLabelProvider.from_provider(self.env.labels)
        sampled_non_relevant = self._provider_sample(self.env.unlabeled)
        for ins_key in sampled_non_relevant:
            temp_labels.set_labels(ins_key, self.neg_label)
        train_set = self.env.combine(sampled_non_relevant, self.env.labeled)
        self.classifier.fit_provider(train_set, temp_labels)

    def _predict(self, provider: il.InstanceProvider[IT, KT, DT, np.ndarray, RT]) -> Tuple[Sequence[KT], np.ndarray]:
        raw_probas = self.classifier.predict_proba_provider_raw(provider)
        keys, matrix = raw_proba_chainer(raw_probas)
        return keys, matrix

    def _sample(self, keys: Sequence[KT], matrix: np.ndarray) -> il.InstanceProvider[IT, KT, DT, np.ndarray, RT]:
        pass

    def __next__(self) -> IT:
        if self.batch_it == 0:
            self._temp_augment()
            keys, matrix = self._predict(self.env.unlabeled)

            

                
        