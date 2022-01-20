import random
from typing import FrozenSet, Generic, Optional, Sequence, Tuple, Dict

import instancelib as il
import numpy as np
from instancelib.typehints import DT, KT, LT, RTunlabeled

from allib.utils.func import list_unzip, sort_on

from ..environment.base import AbstractEnvironment
from ..typehints import IT
from ..typehints.typevars import RT, VT
from ..utils.numpy import raw_proba_chainer
from .insclass import ILLabelProbabilityBased, ILMLBased, ILProbabilityBased
from .poolbased import PoolBasedAL
from .selectioncriterion import AbstractSelectionCriterion

def calc_ap_prior_distribution(ranking: Sequence[KT]) -> Sequence[Tuple[KT, float]]:
    ranks, keys = list_unzip(enumerate(ranking))    
    ar_ranks = np.array(ranks)
    N = len(ranking)
    Z = np.sum(np.log(N/ar_ranks))
    pis: np.ndarray = 1.0 / Z * np.log(N/ar_ranks)
    return list(zip(keys, pis.tolist()))
        


class AutoStopLearner(PoolBasedAL[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, RT, LT]):
    rank_history: Dict[int, Dict[KT, int]]
    distribution_history: Dict[int, Dict[KT, float]]
    sampled_sets : Dict[int, il.InstanceProvider[IT, KT, DT, VT, RT]]
    current_sample: il.InstanceProvider[IT, KT, DT, VT, RT]

    def __init__(self, 
                 classifier: il.AbstractClassifier[IT, KT, DT, VT, RT, LT, np.ndarray, np.ndarray],
                 pos_label: LT, 
                 neg_label: LT, 
                 k_sample: int,
                 batch_size: int,  
                 *_, identifier: Optional[str] = None, **__) -> None:
        super().__init__(*_, identifier=identifier, **__)
        self.classifier = classifier
        self.it = 0
        self.batch_it = 0
        self.trained: bool = False
        self.k_sample = k_sample
        self.pos_label = pos_label
        self.neg_label = neg_label
        self.batch_size = batch_size
        self.rank_history = dict()
        self.distribution_history = dict()
        self.sampled_sets = dict()
        self.current_sample = il.DataPointProvider([])

    def __call__(self, environment: AbstractEnvironment[IT, KT, DT, VT, RT, LT]) -> PoolBasedAL[IT, KT, DT, VT, RT, LT]:
        super().__call__(environment)
        self.classifier.set_target_labels(self.env.labels.labelset)
    
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

    def _rank(self, provider: il.InstanceProvider[IT, KT, DT, np.ndarray, RT]) -> Sequence[Tuple[KT, float]]:
        keys, matrix = self._predict(provider)
        pos_column = self.classifier.get_label_column_index(self.pos_label)
        prob_vector = matrix[:,pos_column]
        floats: Sequence[float] = prob_vector.tolist()
        zipped = list(zip(keys, floats))
        ranking = sort_on(1, zipped)
        return ranking
    
    def inclusion_probability(self, key: KT) -> float:
        dist_h = self.distribution_history
        pi = 1.0 - np.product(
            [((1.0 - dist_h[t][key]) ** len(self.sampled_sets[t])) for t in dist_h])
        return pi

    def second_order_probability(self, key_a: KT, key_b: KT) -> float:
        dist_h = self.distribution_history
        min_part = 1.0 - np.product([((1.0 - dist_h[t][key_a] - dist_h[t][key_b]) ** len(self.sampled_sets[t])) for t in dist_h])
        pij = self.inclusion_probability(key_a) + self.inclusion_probability(key_b) - min_part
        return pij

    def _sample(self, ranking: Sequence[Tuple[KT, float]]) -> il.InstanceProvider[IT, KT, DT, np.ndarray, RT]:
        ranked_keys, _ = list_unzip(ranking)
        distribution = calc_ap_prior_distribution(ranked_keys)
        keys, probs = list_unzip(distribution)
        sample = random.choices(keys, weights=probs, k=self.k_sample)
        return self.env.create_bucket(sample)

    def __next__(self) -> IT:
        if self.batch_it == 0:
            self._temp_augment()
            ranking = self._rank(self.env.dataset)
            self.distribution_history[self.it] = calc_ap_prior_distribution(ranking)
            sample = self._sample(ranking)
            self.sampled_sets[self.it] = sample
            self.current_sample = self.env.create_bucket(sample)
        if not self.current_sample.empty:
            elem = next(iter(self.sampled_sets.values()))
        self.
        if self.batch_it == self.batch_size:
           self.batch_size = self.batch_size + round(self.batch_size / 10)
           self.batch_it = 0
           self.it+= 1
        else:
           self.batch_size
            
            
            

            

                
        