from re import M
from typing import Any, Generic, Tuple

import numpy as np
import numpy.typing as npt

from ..activelearning import ActiveLearner
from ..activelearning.autostop import AutoStopLearner
from ..typehints import DT, IT, KT, LT, RT, VT
from .base import AbstractEstimator, Estimate

from numba import njit


def calc_fo_mask(learner: AutoStopLearner[Any, Any, Any, Any, Any, Any], it: int) -> npt.NDArray[Any]:
    labeled = learner.cumulative_sampled[it]
    fo_mask = np.array([int(k in labeled) for k in learner.key_seq]).reshape((1,-1))
    return fo_mask

def calc_so_mask(fo_mask: npt.NDArray[Any], big_n: int) -> npt.NDArray[Any]:
    mat = np.tile(fo_mask, (big_n, 1))
    mat_t = mat.T
    so_mask = mat * mat_t
    return so_mask

def horvitz_thompson_point(learner: AutoStopLearner[Any, Any, Any, Any, Any, Any], 
                           it: int) -> float:
    
    fo_mask = calc_fo_mask(learner, it)
    fo_prob = learner.fo_inclusion_probabilities(it)
    ys = learner.label_vector[it]
    estimate = np.sum(fo_mask * (ys / fo_prob))
    return estimate

def horvitz_thompson_var1(
        learner: AutoStopLearner[Any, Any, Any, Any, Any, Any],
        it: int) -> float:
    fo_mask = calc_fo_mask(learner, it)    
    big_n = len(learner.env.dataset)
    so_mask = calc_so_mask(fo_mask, big_n)
    
    ys = learner.label_vector[it]
    fo_pi = learner.fo_inclusion_probabilities(it)
    so_pij = learner.so_inclusion_probabilities(it)
    
    part1 = 1.0 / fo_pi ** 2 - 1.0 / fo_pi
    yi_2 = ys ** 2
    
    # 1/(pi_i*pi_j) - 1/pi_ij
    M = np.tile(fo_pi, (big_n, 1))
    MT = M.T
    part2 = 1.0 / (M * MT) - 1.0 / so_pij
    np.fill_diagonal(part2, 0.0)  # set diagonal values to zero, because summing part2 do not include diagonal values

    #  y_i * y_j
    M = np.tile(ys, (big_n, 1))
    MT = M.T
    yi_yj = M*MT
    
    variance = np.sum(fo_mask * part1*yi_2) + np.sum(so_mask *part2*yi_yj)
    return variance

def horvitz_thompson_var2(learner: AutoStopLearner[Any, Any, Any, Any, Any, Any], it: int) -> float:
    fo_mask = calc_fo_mask(learner, it)
    big_n = len(learner.env.dataset)
    v = len(learner.cumulative_sampled[it])
    ys = learner.label_vector[it]
    total = horvitz_thompson_point(learner, it)
    fo_prob = learner.fo_inclusion_probabilities(it)

    # (v * y_i / pi_i - total)**2
    variance2 = (v * ys / fo_prob - total) ** 2
    variance2 = (big_n - v) / big_n / v / (v - 1) * np.sum(fo_mask*variance2)
    return variance2



class HorvitzThompsonVar1(AbstractEstimator[IT, KT, DT, VT, RT, LT], 
                      Generic[IT, KT, DT, VT, RT, LT]):
    def __init__(self) -> None:
        super().__init__()
        self.it = -1
        self.estimate = Estimate(float("nan"), float("nan"), float("nan"))
    

    def calculate(self, learner: AutoStopLearner, it: int) -> Estimate:
        point = horvitz_thompson_point(learner, it)
        variance = horvitz_thompson_var1(learner, it)
        error = np.sqrt(variance)
        estimate = Estimate(point, point - error, point + error)
        return estimate

    def __call__(self, learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT) -> Estimate:
        assert isinstance(learner, AutoStopLearner)
        if learner.cumulative_fo:
            it = min(max(learner.cumulative_fo.keys()), learner.it)
            if it > 1 and self.it != it:
                self.estimate = self.calculate(learner, it)
        return self.estimate
        
class HorvitzThompsonVar2(HorvitzThompsonVar1[IT, KT, DT, VT, RT, LT], 
                      Generic[IT, KT, DT, VT, RT, LT]):
    def calculate(self, learner: AutoStopLearner, it: int) -> Estimate:
        point = horvitz_thompson_point(learner, it)
        variance = horvitz_thompson_var2(learner, it)
        error = np.sqrt(variance)
        estimate = Estimate(point, point - error, point + error)
        return estimate

    

class HorvitzThompsonLoose(HorvitzThompsonVar1[IT, KT, DT, VT, RT, LT], 
                      Generic[IT, KT, DT, VT, RT, LT]):
    def calculate(self, learner: AutoStopLearner, it: int) -> Estimate:
        point = horvitz_thompson_point(learner, it)
        estimate = Estimate(point, point, point)
        return estimate
