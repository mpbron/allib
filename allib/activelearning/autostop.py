import random
from typing import Deque, FrozenSet, Generic, Optional, Sequence, Tuple, Dict

import instancelib as il
import collections
import numpy as np
from instancelib.typehints import DT, KT, LT, RT, VT

from allib.utils.func import list_unzip, sort_on

from ..environment.base import AbstractEnvironment
from ..typehints import IT
from ..utils.numpy import raw_proba_chainer
from .insclass import ILLabelProbabilityBased, ILMLBased, ILProbabilityBased
from .poolbased import PoolBasedAL
from .selectioncriterion import AbstractSelectionCriterion

def calc_ap_prior_distribution(ranking: Sequence[Tuple[KT, float]]) -> Sequence[Tuple[KT, float]]:
    keys, _ = list_unzip(ranking)
    N = len(ranking)
    ranks = np.array(range(1,(N+1)))   
    Z = np.sum(np.log(N/ranks))
    pis: np.ndarray = 1.0 / Z * np.log(N/ranks)
    return list(zip(keys, pis.tolist()))
        


class AutoStopLearner(PoolBasedAL[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]):
    rank_history: Dict[int, Dict[KT, int]]
    distributions: Dict[int, Dict[KT, float]]
    sampled_sets : Dict[int, Sequence[KT]]
    current_sample: Deque[KT]

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
        self.distributions = dict()
        self.sampled_sets = dict()
        self.current_sample = list()

    def __call__(self, environment: AbstractEnvironment[IT, KT, DT, VT, RT, LT]) -> PoolBasedAL[IT, KT, DT, VT, RT, LT]:
        super().__call__(environment)
        self.classifier.set_target_labels(self.env.labels.labelset)
        return self

    def update_ordering(self) -> bool:
        return True

    def _provider_sample(self, provider: il.InstanceProvider[IT, KT, DT, np.ndarray, RT]) -> il.InstanceProvider[IT, KT, DT, np.ndarray, RT]:
        k_sample = min(self.k_sample, len(provider))
        sampled_keys = random.sample(provider.key_list, k_sample)
        sampled_provider = self.env.create_bucket(sampled_keys)
        return sampled_provider

    def _temp_augment_and_train(self):
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
        dist_h = self.distributions
        pi = 1.0 - np.product(
            [((1.0 - dist_h[t][key]) ** len(self.sampled_sets[t])) for t in dist_h])
        return pi

    def second_order_probability(self, key_a: KT, key_b: KT) -> float:
        dist_h = self.distributions
        min_part = 1.0 - np.product([((1.0 - dist_h[t][key_a] - dist_h[t][key_b]) ** len(self.sampled_sets[t])) for t in dist_h])
        pij = self.inclusion_probability(key_a) + self.inclusion_probability(key_b) - min_part
        return pij

    def _sample(self, distribution: Sequence[Tuple[KT, float]]) -> Sequence[KT]:
        keys, probs = list_unzip(distribution)
        sample = random.choices(keys, weights=probs, k=self.batch_size)
        return sample

    def __next__(self) -> IT:
        if not self.current_sample:
            self._temp_augment_and_train()
            ranking = self._rank(self.env.dataset)
            distribution = calc_ap_prior_distribution(ranking)
            sample = self._sample(distribution)
            
            # Store all data for later analysis
            self.distributions[self.it] = distribution
            self.sampled_sets[self.it] = tuple(sample)
            self.current_sample = collections.deque(sample)
            self.batch_size += round(self.batch_size / 10)
            self.it+= 1

        while self.current_sample:
            ins_key = self.current_sample.popleft()
            if ins_key not in self.env.labeled:
                return self.env.dataset[ins_key]
        if not self.env.unlabeled.empty:
            return self.__next__()
        raise StopIteration()
            
            
            

            

                
        