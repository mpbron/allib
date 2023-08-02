from __future__ import annotations

import collections
import logging
import os
from typing import (
    Any,
    Deque,
    Dict,
    FrozenSet,
    Generic,
    Mapping,
    Optional,
    Sequence,
    Set,
    Tuple,
)
import warnings

import math
import numpy as np
import numpy.typing as npt

import pandas as pd

from allib.activelearning.base import ActiveLearner
from allib.estimation.base import Estimate  # type: ignore
from ..activelearning.autotarensemble import AutoTARFirstMethod
from ..activelearning.base import ActiveLearner
from ..activelearning.estimator import Estimator
from ..utils.func import (
    intersection,
    not_in_supersets,
    powerset,
    union,
)
from .base import AbstractEstimator, Estimate
from instancelib.utils.func import value_map
import instancelib as il

from ..typehints import IT, KT, DT, RT, LT, VT

LOGGER = logging.getLogger(__name__)

EPSILON = 0.00001


class QuantEstimator(
    AbstractEstimator[IT, KT, DT, VT, RT, LT], Generic[IT, KT, DT, VT, RT, LT]
):
    previously_labeled: FrozenSet[KT]

    def __init__(self, nstd: float):
        self.est = Estimate.empty()
        self.nstd = nstd
        self.previously_labeled = frozenset()

    def __call__(
        self, learner: ActiveLearner[IT, KT, DT, VT, RT, LT], label: LT
    ) -> Estimate:
        if hasattr(learner, "classifier") and isinstance(learner.classifier, il.AbstractClassifier):  # type: ignore
            clf: il.AbstractClassifier[Any, Any, Any, Any, Any, LT, npt.NDArray[np.int64], npt.NDArray[np.float64]] = learner.classifier  # type: ignore
            labeled = frozenset(learner.env.labeled)
            if labeled.difference(self.previously_labeled):
                # If there are new instance labeled since last estimate
                self.previously_labeled = labeled
                scores = {
                    key: dict(pred)[label]
                    for key, pred in clf.predict_proba(learner.env.dataset)
                }
                unknown_ps = sum([scores[k] for k in learner.env.unlabeled])
                known_ps = sum([scores[k] for k in learner.env.labeled])
                unknown_var = sum(
                    [scores[k] * (1 - scores[k]) for k in learner.env.unlabeled]
                )
                # Calculate recall
                all_var = sum([s * (1 - s) for s in scores.values()])
                est_recall = (known_ps) / (known_ps + unknown_ps)
                est_var = (known_ps**2 / (known_ps + unknown_ps) ** 4 * all_var) + (
                    1 / (known_ps + unknown_ps) ** 2 * (all_var - unknown_var)
                )
                est_recall_ci = self.nstd * np.sqrt(est_var)

                def ensure_within_interval(
                    x: float, min_e: float = EPSILON, max_u: float = 1.0
                ) -> float:
                    # Ensure that estimates are est >= EPSILON (0.0001) and est <= 1.0)
                    return max(min_e, min(max_u, x))

                est_recall_bounded = ensure_within_interval(est_recall)
                est_recall_ci_upper = ensure_within_interval(est_recall + est_recall_ci)
                est_recall_ci_lower = ensure_within_interval(est_recall - est_recall_ci)

                # Revert back to count estimates
                n = learner.env.labels.document_count(label)
                # LB: Because a higher recall percentage results in lower count estimate
                lower_bound = n / est_recall_ci_upper
                point = n / est_recall_bounded
                # LB: Because a lower recall percentage results in higher count estimate
                upper_bound = n / est_recall_ci_lower
                self.est = Estimate(point, lower_bound, upper_bound)
        return self.est
