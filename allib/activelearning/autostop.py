import random
from typing import Deque, FrozenSet, Generic, Optional, Sequence, Tuple, Dict

import instancelib as il
import collections
import numpy as np
from instancelib.typehints import DT, KT, LT, RT, VT
from allib.activelearning.autotar import AutoTarLearner

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
        


class AutoStopLearner(AutoTarLearner[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]):
    distributions: Dict[int, Dict[KT, float]]
    
    def __init__(self, 
                 classifier: il.AbstractClassifier[IT, KT, DT, VT, RT, LT, np.ndarray, np.ndarray],
                 pos_label: LT, 
                 neg_label: LT, 
                 k_sample: int,
                 batch_size: int,  
                 *_, identifier: Optional[str] = None, **__) -> None:
        super().__init__(classifier, pos_label, neg_label, k_sample, batch_size, identifier=identifier)
        self.distributions = dict()   

    def inclusion_probability(self, key: KT, t_max: int) -> float:
        dist_h = self.distributions
        pi = 1.0 - np.product(
            [((1.0 - dist_h[t][key]) ** len(self.sampled_sets[t])) for t in dist_h if t <= t_max])
        return pi

    def second_order_probability(self, key_a: KT, key_b: KT, t_max: int) -> float:
        dist_h = self.distributions
        min_part = 1.0 - np.product([((1.0 - dist_h[t][key_a] - dist_h[t][key_b]) ** len(self.sampled_sets[t])) for t in dist_h if t <= t_max])
        pij = self.inclusion_probability(key_a, t_max) + self.inclusion_probability(key_b, t_max) - min_part
        return pij

    def _sample(self, distribution: Sequence[Tuple[KT, float]]) -> Sequence[KT]:
        keys, probs = list_unzip(distribution)
        sample = random.choices(keys, weights=probs, k=self.batch_size)
        return sample

    def horvitz_thompson(self, it: int) -> float:
        sample = self.sampled_sets[it]
        unique = frozenset(sample)
        ys = np.array([int(self.pos_label in self.env.labels.get_labels(key)) for key in unique])
        ps = np.array([self.inclusion_probability(key, it) for key in unique])
        estimate = np.sum(ys / ps)
        return estimate        

    def update_sample(self) -> Deque[KT]:
        if not self.current_sample:
            self._temp_augment_and_train()
            ranking = self._rank(self.env.dataset)
            distribution = calc_ap_prior_distribution(ranking)
            sample = self._sample(distribution)
            
            # Store all data for later analysis
            self.distributions[self.it] = dict(distribution)
            self.rank_history[self.it] = self._to_history(ranking)
            self.sampled_sets[self.it] = tuple(sample)
            self.current_sample = collections.deque(sample)
            self.batch_size += round(self.batch_size / 10)
            self.it+= 1
        return self.current_sample
            
            
            

            

                
        