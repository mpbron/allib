from typing import Any, Dict, FrozenSet, Generic, List, Tuple

import numpy as np
import pandas as pd

from ..activelearning import ActiveLearner
from ..activelearning.estimator import Estimator
from ..typehints.typevars import DT, KT, LT, RT, VT
from ..utils.func import all_subsets, intersection, powerset, not_in_supersets, union

import functools


def convert_df(df: pd.DataFrame):
    table = df.filter(regex="learner|positive").filter(regex="^((?!-positive).)*$").values.tolist()
    def match_chars(ar: List[int]) -> str:
        string = "ABC+"
        zipped = zip(ar, string)
        filtered = "".join([y for (x,y) in zipped if x == 1])
        if "+" not in filtered:
            return filtered + "-"
        return filtered
    colnames = list(map(match_chars, table))
    values = np.array(df["count"].values.tolist()).reshape((1,14)) # type: ignore
    new_row = pd.DataFrame(data=values, columns=colnames)
    formatted = new_row[["A+", "B+", "C+", "AB+", "AC+", "BC+", "ABC+",
                         "A-", "B-", "C-", "AB-", "AC-", "BC-", "ABC-"]]
    return formatted
# %%
class TableCollector(Generic[LT]):
    def __init__(self, pos_label: LT):
        self.dfs = list()
        self.pos_label = pos_label

    def get_contingency_sets(self, 
                         estimator: Estimator[Any, KT, Any, Any, Any, LT], 
                         label: LT) -> Dict[FrozenSet[int], FrozenSet[KT]]:
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
    
    def get_occasion_history(self, 
                             estimator: Estimator[Any, Any, Any, Any, Any, LT], 
                             label: LT) -> pd.DataFrame:
        contingency_sets_pos = self.get_contingency_sets(estimator, label)
        contingency_sets_neg = self.get_contingency_sets_negative(estimator, label)
        learner_keys = union(*contingency_sets_pos.keys())
        
        rasch_pos_func = functools.partial(
            self.rasch_row, all_learners=learner_keys, positive = True)
        rasch_neg_func = functools.partial(
            self.rasch_row, all_learners=learner_keys, positive = False)
        
        
        rows_pos = list(map(rasch_pos_func, contingency_sets_pos.items()))
        rows_neg = list(map(rasch_neg_func, contingency_sets_neg.items()))

        rows = dict(enumerate(rows_pos + rows_neg))
        
        df = pd.DataFrame.from_dict(# type: ignore
            rows, orient="index")
        return df

    def get_contingency_sets_negative(self, 
                                      estimator: Estimator[Any, KT, Any, Any, Any, LT], 
                                      label: LT) -> Dict[FrozenSet[int], FrozenSet[KT]]:
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

    @staticmethod
    def rasch_row(combination: Tuple[FrozenSet[int], FrozenSet[Any]], 
              all_learners: FrozenSet[int],
              positive: bool) -> Dict[str, int]:
        learner_set, instances = combination
        learner_cols = {
            f"learner_{learner_key}": int(learner_key in learner_set) 
            for learner_key in all_learners
        }
        count_col = {"count": len(instances)}
        positive_col = {"positive": int(positive)}
        interaction_cols = {
            f"h{i-1}": len(all_subsets(learner_set, i, i)) 
            for i in range(2, len(all_learners))
        }
        pos_learner_cols = {
            f"learner_{learner_key}-positive": (
                int(learner_key in learner_set) if positive else 0)
            for learner_key in all_learners
        }
        interaction_pos_cols = {
            f"h{i-1}-positive": (
                len(all_subsets(learner_set, i, i)) if positive else 0)
            for i in range(2, len(all_learners))
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

    def __call__(self, learner: ActiveLearner):
        if isinstance(learner, Estimator):
            df = self.get_occasion_history(learner, self.pos_label)
            self.dfs.append(df)

    @property
    def compact(self) -> pd.DataFrame:
        compact = pd.concat(map(convert_df, self.dfs), ignore_index=True)
        return compact # type: ignore