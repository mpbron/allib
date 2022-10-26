import collections
import functools
from typing import Any, Deque, FrozenSet, Generic, Mapping, Tuple

import pandas as pd
import numpy as np
import numpy.typing as npt
from instancelib.typehints import DT, KT, LT, RT, VT

from string import ascii_uppercase
from ..activelearning import ActiveLearner
from ..activelearning.estimator import Estimator
from ..utils.func import all_subsets, intersection, not_in_supersets, powerset, union
from .base import AbstractEstimator, Estimate
from .rasch import EMRaschCombined
from .rasch_multiple import pos_rasch_numpy
from ..analysis.statistics import EstimationModelStatistics
from ..typehints import IT


def log_linear_estimate(df: pd.DataFrame, dataset_size: int, multinomial_size: int) -> Tuple[Estimate, EstimationModelStatistics]:
    df_formatted = df.sort_values([col for col in ascii_uppercase if col in df.columns])
    df_formatted.insert(0, "intercept", 1)
    design_mat = (
        df_formatted.loc[:, df_formatted.columns != 'count'] # type: ignore
            .values) # type: ignore
    design_mat = design_mat.astype("float64")
    obs_counts: npt.NDArray[Any] = df_formatted["count"].values # type: ignore
    total_found: int = int(np.sum(obs_counts))
    
    point, low, up, beta, mfit, deviance, estimates = pos_rasch_numpy(
        design_mat, obs_counts, total_found, max_it=multinomial_size)
    estimate = Estimate(point, low, up)
    stats = EstimationModelStatistics(beta, mfit, deviance, estimates)
    return estimate, stats

def log_linear_em_estimate(df: pd.DataFrame, dataset_size: int, multinomial_size: int) -> Tuple[Estimate, EstimationModelStatistics]:
    df_formatted = df.sort_values([col for col in ascii_uppercase if col in df.columns])
    df_formatted.insert(0, "intercept", 1)
    design_mat = (
        df_formatted.loc[:, df_formatted.columns != 'count'] # type: ignore
            .values) # type: ignore
    design_mat = design_mat.astype("float64")
    obs_counts: npt.NDArray[Any] = df_formatted["count"].values # type: ignore
    total_found: int = int(np.sum(obs_counts))
    
    point, low, up, beta, mfit, deviance, estimates = pos_rasch_numpy(
        design_mat, obs_counts, total_found, max_it=multinomial_size)
    estimate = Estimate(point, low, up)
    stats = EstimationModelStatistics(beta, mfit, deviance, estimates)
    return estimate, stats

def learner_key(l: int) -> str:
    return ascii_uppercase[l] 

def learnercombi_string(lset: FrozenSet[int]) -> str:
    strlset = "".join([learner_key(l) for l in sorted(lset)])
    return strlset

def get_contingency_sets_negative(estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                                  label: LT) -> Mapping[FrozenSet[int], FrozenSet[KT]]:
    learner_sets = {
            learner_key: (
                frozenset(learner.env.labeled).
                    difference(
                        learner.env.labels.get_instances_by_label(label))
            )
        for learner_key, learner in enumerate(estimator.learners)
    }
    key_combinations = powerset(range(len(estimator.learners)))
    result = {
        combination: intersection(*[learner_sets[key] for key in combination])
        for combination in key_combinations
        if len(combination) >= 1
    }
    filtered_result = not_in_supersets(result)
    return filtered_result

def get_contingency_sets(estimator: Estimator[Any, KT, Any, Any, Any, LT], 
                         label: LT) -> Mapping[FrozenSet[int], FrozenSet[KT]]:
    learner_sets = {
        learner_key: learner.env.labels.get_instances_by_label(
            label).intersection(learner.env.labeled)
        for learner_key, learner in enumerate(estimator.learners)
    }
    key_combinations = powerset(range(len(estimator.learners)))
    result = {
        combination: intersection(*[learner_sets[key] for key in combination])
        for combination in key_combinations
        if len(combination) >= 1
    }
    filtered_result = not_in_supersets(result)
    return filtered_result

    

class LogLinear(AbstractEstimator[IT, KT, DT, VT, RT, LT],
        Generic[IT, KT, DT, VT, RT, LT]):

    estimates: Deque[Estimate]
    model_info: Deque[EstimationModelStatistics]
    dfs: Deque[pd.DataFrame]

    def __init__(self, multinomial_size: int = 2000):
        super().__init__()
        self.estimates = collections.deque()
        self.dfs = collections.deque()
        self.model_info = collections.deque()
        self.multinomial_size = multinomial_size

    def __call__(self, learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT) -> Estimate:
        assert isinstance(learner, Estimator)
        return self.calculate_estimate(learner, label)

    def calculate_estimate(self,
                           estimator: Estimator[Any, KT, DT, VT, RT, LT],
                           label: LT) -> Estimate:
        dataset_size = len(estimator.env.dataset)
        df = self.get_design_matrix(estimator, label)
        if not self.dfs or not self.dfs[-1].equals(df):
            self.dfs.append(df)
            est, stats = log_linear_estimate(df, dataset_size, self.multinomial_size)
            self.estimates.append(est)
            self.model_info.append(stats)
        return self.estimates[-1]
    
    
    @classmethod
    def design_matrix_row(cls,
                          combination: Tuple[FrozenSet[int], FrozenSet[Any]],
                          all_learners: FrozenSet[int]) -> Mapping[str, int]:
        learner_set, instances = combination
        learner_cols = {
            learner_key(l): int(l in learner_set) 
            for l in all_learners
        }
        count_col = {"count": len(instances)}
        
        all_possible_interactions = all_subsets(all_learners, 2 , len(all_learners) -1)
        all_present_interactions = all_subsets(learner_set, 2 , len(all_learners) - 1)
        interaction_cols = {
            learnercombi_string(lset): int(lset in all_present_interactions)
            for lset in all_possible_interactions
        }
        final_row = {
            **learner_cols,
            **interaction_cols,
            **count_col
        }
        
        return final_row
    
    @classmethod
    def get_design_matrix(cls, 
                          estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                          label: LT) -> pd.DataFrame:
        contingency_sets_pos = get_contingency_sets(estimator, label)
        learner_keys = union(*contingency_sets_pos.keys())
        
        pos_func = functools.partial(
            cls.design_matrix_row, all_learners=learner_keys)
        rows = dict(enumerate(map(pos_func, contingency_sets_pos.items())))
        df = pd.DataFrame.from_dict(rows, orient="index")
        return df

    
    


class PosNegLogLinear(LogLinear):

    @classmethod
    def pos_neg_row(cls,
                    combination: Tuple[FrozenSet[int], FrozenSet[Any]],
                    all_learners: FrozenSet[int], positive: bool) -> Mapping[str, int]:
        learner_set, _ = combination
        standard_cols = cls.design_matrix_row(combination, all_learners)
        positive_col = {"positive": int(positive)}
        all_possible_interactions = all_subsets(all_learners, 2 , len(all_learners) -1)
        all_present_interactions = all_subsets(learner_set, 2 , len(all_learners) - 1)
        pos_learner_cols = {
            f"{learner_key(l)}_pos": (int(l in learner_set)  if positive else 0)
            for l in all_learners
        }
        interaction_pos_cols = {
            f"{learnercombi_string(lset)}": (int(lset in all_present_interactions) if positive else 0)
            for lset in all_possible_interactions
        }
        final_row = {
            **standard_cols,
            **positive_col,
            **pos_learner_cols,
            **interaction_pos_cols
        }
        return final_row

    @classmethod
    def get_design_matrix(cls, 
                          estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                          label: LT) -> pd.DataFrame:
        contingency_sets_pos = get_contingency_sets(estimator, label)
        contingency_sets_neg = get_contingency_sets_negative(estimator, label)
        learner_keys = union(*contingency_sets_pos.keys())
        
        pos_func = functools.partial(
            cls.pos_neg_row, all_learners=learner_keys, positive = True)
        
        neg_func = functools.partial(
            cls.pos_neg_row, all_learners=learner_keys, positive = False)
                
        rows_pos = list(map(pos_func, contingency_sets_pos.items()))
        rows_neg = list(map(neg_func, contingency_sets_neg.items()))
        rows = dict(enumerate(rows_pos + rows_neg))
        
        df = pd.DataFrame.from_dict(rows, orient="index")
        return df

    