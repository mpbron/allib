from __future__ import annotations
from allib.utils.func import all_subsets, intersection, list_unzip3, union
from allib.activelearning.base import ActiveLearner

import os
from abc import ABC, abstractmethod
from typing import (Any, Deque, Dict, FrozenSet, Generic, Iterable, List,
                    Optional, Sequence, Tuple, TypeVar)

import numpy as np  # type: ignore
import pandas as pd  # type: ignore

from ..activelearning.estimator import Estimator

from .base import AbstractEstimator
from .rcapture import AbundanceEstimator, _check_R
from ..utils.func import powerset, not_in_supersets

try:
    import rpy2.robjects as ro  # type: ignore
    from rpy2.robjects import pandas2ri  # type: ignore
    from rpy2.robjects.conversion import localconverter  # type: ignore
    from rpy2.robjects.packages import importr  # type: ignore
except ImportError:
    R_AVAILABLE = False
else:
    R_AVAILABLE = True

DT = TypeVar("DT")
VT = TypeVar("VT")
KT = TypeVar("KT")
LT = TypeVar("LT")
RT = TypeVar("RT")
LVT = TypeVar("LVT")
PVT = TypeVar("PVT")
_T = TypeVar("_T")
_U = TypeVar("_U")


class RaschEstimator(AbundanceEstimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "RaschEstimator"
    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "rasch_estimate_bootstrap.R")
        R["source"](r_script_file)

    def get_occasion_history(self, 
                             estimator: Estimator[KT, DT, VT, RT, LT], 
                             label: LT) -> pd.DataFrame:
        contingency_sets = self.get_contingency_sets(estimator, label)
        learner_keys = union(*contingency_sets.keys())
        max_len = len(learner_keys)
        rows = {i:
            {
                **{
                    f"learner_{learner_key}": int(learner_key in combination) 
                    for learner_key in learner_keys
                },  
                **{
                    "count": len(instances),
                    "h1": len(all_subsets(combination, 2, max_len - 1))
                }
            }
            for (i, (combination, instances)) in enumerate(contingency_sets.items())
        }
        df = pd.DataFrame.from_dict(# type: ignore
            rows, orient="index")
        self.matrix_history.append(df)
        return df

    def calculate_abundance_R(self, estimator: Estimator[KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df = self.get_occasion_history(estimator, label)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
            abundance_r = ro.globalenv["rasch.single"]
            r_df = abundance_r(df_r)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    
    def calculate_abundance(self, 
                            estimator: Estimator[KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        estimate_missing = res_df.values[0,0]
        estimate_error = res_df.values[0,1]
        total_found = estimator.env.labels.document_count(label)
        return (total_found + estimate_missing), estimate_error * 2

    def all_estimations(self, estimator: Estimator[KT, DT, VT, RT, LT], label: LT) -> Sequence[Tuple[str, float, float]]:
        return []

    def __call__(self, learner: ActiveLearner[KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0, 0.0
        abundance, error = self.calculate_abundance(learner, label)
        return abundance, abundance-error, abundance + error

class NonParametricRasch(RaschEstimator[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "RaschNonParamatric"
    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "rasch_estimate_bootstrap.R")
        R["source"](r_script_file)

    

    def calculate_abundance_R(self, estimator: Estimator[KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df = self.get_occasion_history(estimator, label)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
            abundance_r = ro.globalenv["rasch.nonparametric"]
            r_df = abundance_r(df_r)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_estimate(self, 
                            estimator: Estimator[KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        horizon = res_df.values[0,0]
        horizon_lowerbound = res_df.values[0,1]
        horizon_upperbound = res_df.values[0,2]
        return horizon, horizon_lowerbound, horizon_upperbound
    
    def __call__(self, 
        learner: ActiveLearner[KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0, 0.0
        estimate, lower_bound, upper_bound = self.calculate_estimate(learner, label)
        return estimate, lower_bound, upper_bound

class ParametricRasch(NonParametricRasch[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "RaschParametric"
    def calculate_abundance_R(self, estimator: Estimator[KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df = self.get_occasion_history(estimator, label)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_r = ro.conversion.py2rpy(df)
            abundance_r = ro.globalenv["rasch.parametric"]
            r_df = abundance_r(df_r)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

class EMRasch(NonParametricRasch[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "EMRasch"
    def __init__(self, neg_label: LT) -> None:
        super().__init__()
        self.neg_label = neg_label

    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "rasch_em.R")
        R["source"](r_script_file)

    def calculate_abundance_R(self, estimator: Estimator[KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df_pos = self.get_occasion_history(estimator, label)
        df_neg = self.get_occasion_history(estimator, self.neg_label)
        dataset_size = len(estimator.env.dataset)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_pos_r = ro.conversion.py2rpy(df_pos)
            df_neg_r = ro.conversion.py2rpy(df_neg)
            abundance_r = ro.globalenv["rasch.em"]
            r_df = abundance_r(df_pos_r, df_neg_r, dataset_size)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_estimate(self, 
                            estimator: Estimator[KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        horizon = res_df.values[0,0]
        return horizon, horizon, horizon
    
    def __call__(self, 
        learner: ActiveLearner[KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0, 0.0
        estimate, lower_bound, upper_bound = self.calculate_estimate(learner, label)
        return estimate, lower_bound, upper_bound

class EMRaschCombined(NonParametricRasch[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    name = "EMRaschCombined"
    def __init__(self, neg_label: LT) -> None:
        super().__init__()
        self.neg_label = neg_label

    def get_occasion_history(self, 
                             estimator: Estimator[KT, DT, VT, RT, LT], 
                             label: LT) -> pd.DataFrame:
        contingency_sets_pos = self.get_contingency_sets(estimator, label)
        contingency_sets_neg = self.get_contingency_sets(estimator, self.neg_label)
        learner_keys = union(*contingency_sets_pos.keys())
        max_len = len(learner_keys)
        rows_pos = {i:
            {
                **{
                    f"learner_{learner_key}": int(learner_key in combination) 
                    for learner_key in learner_keys
                },  
                **{
                    "count": len(instances),
                    "h": 1,
                }
            }
            for (i, (combination, instances)) in enumerate(contingency_sets_pos.items())
        }
        rows_neg = {i:
            {
                **{
                    f"learner_{learner_key}": int(learner_key in combination) 
                    for learner_key in learner_keys
                },  
                **{
                    "count": len(instances),
                    "h": 0
                }
            }
            for (i, (combination, instances)) in enumerate(contingency_sets_neg.items(), len(contingency_sets_pos))
        }
        rows = {**rows_pos, **rows_neg}
        df = pd.DataFrame.from_dict(# type: ignore
            rows, orient="index")
        self.matrix_history.append(df)
        return df

    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, "rasch_em.R")
        R["source"](r_script_file)

    def calculate_abundance_R(self, estimator: Estimator[KT, DT, VT, RT, LT], 
                              label: LT) -> pd.DataFrame:
        df_pos = self.get_occasion_history(estimator, label)
        df_neg = self.get_occasion_history(estimator, self.neg_label)
        dataset_size = len(estimator.env.dataset)
        with localconverter(ro.default_converter + pandas2ri.converter):
            df_pos_r = ro.conversion.py2rpy(df_pos)
            df_neg_r = ro.conversion.py2rpy(df_neg)
            abundance_r = ro.globalenv["rasch.em"]
            r_df = abundance_r(df_pos_r, df_neg_r, dataset_size)
            res_df = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_estimate(self, 
                            estimator: Estimator[KT, DT, VT, RT, LT], 
                            label: LT) -> Tuple[float, float, float]:
        res_df = self.calculate_abundance_R(estimator, label)
        horizon = res_df.values[0,0]
        return horizon, horizon, horizon
    
    def __call__(self, 
        learner: ActiveLearner[KT, DT, VT, RT, LT], label: LT) -> Tuple[float, float, float]:
        if not isinstance(learner, Estimator):
            return 0.0, 0.0, 0.0
        estimate, lower_bound, upper_bound = self.calculate_estimate(learner, label)
        return estimate, lower_bound, upper_bound


    