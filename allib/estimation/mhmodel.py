from __future__ import annotations

import collections
import itertools
import logging
import os
from abc import ABC, abstractmethod
from typing import (
    Any,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
)
import warnings

import numpy as np
import numpy.typing as npt

import pandas as pd  # type: ignore

from ..activelearning.base import ActiveLearner
from ..activelearning.estimator import Estimator
from ..utils.func import (
    all_subsets,
    intersection,
    list_unzip3,
    not_in_supersets,
    powerset,
    union,
)
from .base import AbstractEstimator, Estimate

try:
    import rpy2.robjects as ro  # type: ignore
    from rpy2.robjects import pandas2ri  # type: ignore
    from rpy2.robjects.conversion import localconverter  # type: ignore
    from rpy2.robjects.packages import importr  # type: ignore
except ImportError:
    R_AVAILABLE = False
else:
    R_AVAILABLE = True

from ..typehints import IT, KT, DT, RT, LT, VT

LOGGER = logging.getLogger(__name__)

from ..analysis.statistics import EstimationModelStatistics


def _check_R():
    """Checks if Python <-> R interop is available

    Raises
    ------
    ImportError
        If the interop is not available
    """
    if not R_AVAILABLE:
        raise ImportError("Install rpy2 interop")


class ChaoEstimator(
    AbstractEstimator[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    r_loaded: bool
    name = "Mh Chao Estimator"

    def __init__(self):
        self.matrix_history: Deque[pd.DataFrame] = collections.deque()
        self.contingency_history: Deque[Dict[FrozenSet[int], int]] = collections.deque()
        self.r_loaded = False
        self.df: Optional[pd.DataFrame] = None
        self.est = Estimate.empty()
        self.rfile = "mhmodel.R"
        self.rfunc = "get_abundance"

    def _start_r(self) -> None:
        _check_R()
        R = ro.r
        filedir = os.path.dirname(os.path.realpath(__file__))
        r_script_file = os.path.join(filedir, self.rfile)
        R["source"](r_script_file)

    def get_label_matrix(
        self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT
    ) -> pd.DataFrame:
        rows = {
            ins_key: {
                l_key: ins_key in learner.env.labeled
                for l_key, learner in enumerate(estimator.learners)
            }
            for ins_key in estimator.env.labels.get_instances_by_label(label)
        }
        dataframe = pd.DataFrame.from_dict(rows, orient="index")  # type: ignore
        self.matrix_history.append(dataframe)
        return dataframe

    def get_contingency_list(
        self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT
    ) -> Dict[FrozenSet[int], int]:
        learner_sets = {
            learner_key: learner.env.labels.get_instances_by_label(label).intersection(
                learner.env.labeled
            )
            for learner_key, learner in enumerate(estimator.learners)
        }
        key_combinations = powerset(range(len(estimator.learners)))
        result = {
            combination: len(intersection(*[learner_sets[key] for key in combination]))
            for combination in key_combinations
            if len(combination) >= 1
        }
        self.contingency_history.append(result)
        return result

    def get_matrix(
        self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT
    ) -> npt.NDArray[Any]:
        learner_sets = {
            learner_key: learner.env.labels.get_instances_by_label(label).intersection(
                learner.env.labeled
            )
            for learner_key, learner in enumerate(estimator.learners)
        }
        n_learners = len(learner_sets)
        matrix = np.zeros(shape=(n_learners, n_learners))
        for i, key_a in enumerate(learner_sets):
            instances_a = learner_sets[key_a]
            for j, key_b in enumerate(learner_sets):
                if i != j:
                    instances_b = learner_sets[key_b]
                    intersection = instances_a.intersection(instances_b)
                    matrix[i, j] = len(intersection)
        return matrix

    def calculate_abundance_R(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.r_loaded:
            self._start_r()
            self.r_loaded = True
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with localconverter(ro.default_converter + pandas2ri.converter):
                df_r = ro.conversion.py2rpy(df)
                abundance_r = ro.globalenv[self.rfunc]
                r_df = abundance_r(df_r)
                res_df: pd.DataFrame = ro.conversion.rpy2py(r_df)
        return res_df

    def calculate_abundance(
        self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT
    ) -> Estimate:
        df = self.get_label_matrix(estimator, label)
        old_df = self.df
        if old_df is None or not old_df.equals(df):
            self.df = df
            res_df = self.calculate_abundance_R(df)

            def try_float(val: Any) -> float:
                try:
                    parsed = float(val)
                except ValueError:
                    parsed = float("nan")
                return parsed

            point = try_float(res_df["abundance"][0])
            lower = try_float(res_df["infCL"][0])
            upper = try_float(res_df["supCL"][0])
            self.est = Estimate(point, lower, upper)
        return self.est

    def __call__(
        self, learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT
    ) -> Estimate:
        empty = np.array([])
        stats = EstimationModelStatistics(empty, empty, 0.0, empty)
        if not isinstance(learner, Estimator):
            return Estimate(0.0, 0.0, 0.0)
        estimate = self.calculate_abundance(learner, label)
        return estimate

    def all_estimations(
        self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT
    ) -> Sequence[Tuple[str, float, float]]:
        res_df = self.calculate_abundance_R(estimator, label)
        ok_fit = res_df[res_df.infoFit == 0]
        if len(ok_fit) == 0:
            ok_fit = res_df
        results = ok_fit.values
        names = list(ok_fit.index)
        estimations = list(results[:, 0])
        errors = list(results[:, 1])
        tuples = list(zip(names, estimations, errors))
        return tuples

    def get_contingency_sets(
        self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT
    ) -> Dict[FrozenSet[int], FrozenSet[KT]]:
        learner_sets = {
            learner_key: learner.env.labels.get_instances_by_label(label).intersection(
                learner.env.labeled
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

    def get_occasion_history(
        self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT
    ) -> pd.DataFrame:
        contingency_sets = self.get_contingency_sets(estimator, label)
        learner_keys = union(*contingency_sets.keys())
        rows = {
            i: {
                **{
                    f"learner_{learner_key}": int(learner_key in combination)
                    for learner_key in learner_keys
                },
                **{"count": len(instances)},
            }
            for (i, (combination, instances)) in enumerate(contingency_sets.items())
        }
        df = pd.DataFrame.from_dict(rows, orient="index")  # type: ignore
        return df


class ChaoAlternative(
    ChaoEstimator[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    def __init__(self):
        super().__init__()
        self.rfunc = "get_abundance_eta"
