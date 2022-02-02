from typing import Any, Generic, Tuple

import numpy as np

from ..activelearning import ActiveLearner
from ..activelearning.autostop import AutoStopLearner
from ..typehints import DT, IT, KT, LT, RT, VT
from .base import AbstractEstimator, Estimate


 

def masks(learner: AutoStopLearner[Any, Any, Any, Any, Any, Any], 
          it: int) -> Tuple[np.ndarray, np.ndarray]:
    big_n = len(learner.key_seq)
    sample = frozenset(learner.sampled_sets[it])
    fo_mask = np.array(
        [int(k in sample) for k in learner.key_seq]).reshape((1,-1))
    mat = np.tile(fo_mask, big_n, 1)
    mat_t = mat.T
    so_mask = mat * mat_t
    return fo_mask, so_mask



def horvitz_thompson_var1(
        learner: AutoStopLearner[Any, Any, Any, Any, Any, Any],
        it: int) -> Estimate:
    fo_mask, so_mask = masks(learner, it)
    
    N = len(learner.env.dataset)
    
    sample = learner.sampled_sets[it]
    unique = frozenset(sample)
    ys = learner.label_vector[it]
    ps = learner.fo_inclusion_probabilities(it)
    ss = learner.so_inclusion_probabilities(it)
    
    point_estimate = np.sum(fo_mask * (ys / ps))
    part1 = 1.0 / ps ** 2 - 1.0 / ps
    yi_2 = ys ** 2
    
    # 1/(pi_i*pi_j) - 1/pi_ij
    M = np.tile(ps, (N, 1))
    MT = M.T
    part2 = 1.0 / (M * MT) - 1.0 / ss
    np.fill_diagonal(part2, 0.0)  # set diagonal values to zero, because summing part2 do not include diagonal values

    #  y_i * y_j
    M = np.tile(ys, (N, 1))
    MT = M.T
    yi_yj = M*MT
    

    variance1 = np.sum(fo_mask * part1*yi_2) + np.sum(so_mask *part2*yi_yj)
    estimate = np.sum(ys / ps)
    return estimate  

class HorvitzThompson(AbstractEstimator[LT], Generic[IT, KT, DT, VT, RT, LT]):

    def __call__(self, learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT) -> Estimate:
        assert isinstance(learner, AutoStopLearner)
        point_est = horvitz_thompson(learner, learner.it)
        return Estimate(point_est, point_est, point_est)


