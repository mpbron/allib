import collections
from typing import Any, Deque, FrozenSet, Generic, Mapping, Tuple

import pandas as pd
from instancelib.typehints import DT, KT, LT, RT, VT

from string import ascii_uppercase
from ..activelearning import ActiveLearner
from ..activelearning.estimator import Estimator
from ..utils.func import all_subsets
from .base import Estimate
from .rasch import EMRaschCombined
from .rasch_multiple import ModelStatistics


class FastEMRaschPosNeg(
        EMRaschCombined[KT, DT, VT, RT, LT],
        Generic[KT, DT, VT, RT, LT]):

    estimates: Deque[Estimate]
    model_info: Deque[ModelStatistics]
    dfs: Deque[pd.DataFrame]

    def __init__(self, multinomial_size: int = 2000):
        super().__init__()
        self.estimates = collections.deque()
        self.dfs = collections.deque()
        self.model_info = collections.deque()
        self.multinomial_size = multinomial_size

    def _start_r(self) -> None:
        pass

    def __call__(self, learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT) -> Estimate:
        assert isinstance(learner, Estimator)
        return self.calculate_estimate(learner, label)

    @staticmethod
    def rasch_row(combination: Tuple[FrozenSet[int], FrozenSet[Any]],
                  all_learners: FrozenSet[int],
                  positive: bool) -> Mapping[str, int]:
        def learner_key(l: int) -> str:
            return ascii_uppercase[l] 
        def learnercombi_string(lset: FrozenSet[int]) -> str:
            lset = "".join([learner_key(l) for l in sorted(lset)])
            return lset
       
        learner_set, instances = combination
        learner_cols = {
            learner_key(l): int(l in learner_set) 
            for l in all_learners
        }
        count_col = {"count": len(instances)}
        positive_col = {"positive": int(positive)}
        interaction_cols = {""}
        all_possible_interactions = all_subsets(all_learners, 2 , len(all_learners))
        all_present_interactions = all_subsets(learner_set, 2 , len(all_learners))
        interaction_cols = {
            learnercombi_string(lset): int(lset in all_present_interactions)
            for lset in all_possible_interactions
        }
        pos_learner_cols = {
            f"{learner_key(l)}_pos": (int(l in learner_set)  if positive else 0)
            for l in all_learners
        }
        interaction_pos_cols = {
            f"{learnercombi_string(lset)}": (int(lset in all_present_interactions) if positive else 0)
            for lset in all_possible_interactions
        }
        final_row = {
            **learner_cols,
            **positive_col,
            **pos_learner_cols,
            **interaction_cols,
            **interaction_pos_cols,
            **count_col
        }
        return final_row

    def calculate_estimate(self,
                           estimator: Estimator[Any, KT, DT, VT, RT, LT],
                           label: LT) -> Estimate:
        dataset_size = len(estimator.env.dataset)
        df = self.get_occasion_history(estimator, label)
        if not self.dfs or not self.dfs[-1].equals(df):
            self.dfs.append(df)
            est, stats = rasch_estimate_bf(df, dataset_size)
            self.estimates.append(est)
            self.model_info.append(stats)
        return self.estimates[-1]
