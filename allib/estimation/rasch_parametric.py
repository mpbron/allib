from instancelib.utils.func import all_subsets
import multiprocessing as mp
from typing import Any, Dict, FrozenSet, Generic, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..activelearning.estimator import Estimator
from ..typehints import DT, KT, LT, RT, VT
from .rasch import NonParametricRasch
from .rasch_python import calc_deviance, l2


def l2_format(freq_df: pd.DataFrame, learner_cols: Sequence[str]) -> pd.DataFrame:
    df = freq_df.sort_values(learner_cols)
    df.insert(0, "intercept", 1)
    return df

def glm(design_mat: np.ndarray, counts: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    b0 = np.hstack(
        [
            np.repeat(1, design_mat.shape[1])
        ])
    beta = l2(b0, design_mat, counts, 0)
    mfit = np.exp(design_mat @ beta)
    deviance = calc_deviance(counts, mfit)
    return beta, mfit, deviance

def rasch_estimate(freq_df: pd.DataFrame, 
                   n_dataset: int,
                   proportion: float = 0.1,
                   tolerance: float = 1e-5, 
                   max_it: int = 2000) -> Tuple[float, float, float]:
    


    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(freq_df.filter(regex=r"^learner")
        )
    df_formatted = l2_format(freq_df, learner_cols)
   
    design_mat = (
        df_formatted.loc[:, df_formatted.columns != 'count'] # type: ignore
            .values) # type: ignore
    
    obs_counts: np.ndarray = df_formatted["count"].values # type: ignore
    total_found = np.sum(obs_counts)
    
    beta, mfit, deviance = glm(design_mat, obs_counts)
    estimate =np.exp(beta[0])
    horizon_estimate = total_found + estimate

    fitted: np.ndarray = np.concatenate((np.array([estimate]), mfit))
    p_vals = fitted / np.sum(fitted)

    if not np.isnan(estimate):
        multinomial_fits: np.ndarray = np.random.multinomial(horizon_estimate, p_vals, max_it)
        only_observable_counts: List[List[float]] = multinomial_fits[:,1:].tolist()
        
        workload = [(design_mat, np.array(m_count)) for m_count in only_observable_counts]
        with mp.Pool(7) as pool:
            results = pool.starmap(glm, workload)
        
        estimates = [np.exp(result[0][0]) for result in results]
        
        middle_estimate = np.percentile(estimates, 50)
        low_estimate = np.percentile(estimates, 2.5)
        high_estimate = np.percentile(estimates, 97.5)

        lower_bound = low_estimate
        upper_bound = high_estimate
    
        return middle_estimate, lower_bound, upper_bound
    return horizon_estimate, horizon_estimate, horizon_estimate


class ParametricRaschPython(NonParametricRasch[KT, DT, VT, RT, LT], 
                      Generic[KT, DT, VT, RT, LT]):
    name = "RaschParametric"
    def __init__(self):
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.est = float("nan")
        self.est_low = float("nan")
        self.est_high = float("nan")
    
    def _start_r(self) -> None:
        pass
          
    def calculate_estimate(self, 
                           estimator: Estimator[Any, KT, DT, VT, RT, LT], 
                           label: LT) -> Tuple[float, float, float]:
        pos_count = estimator.env.labels.document_count(label)
        dataset_size = len(estimator.env.dataset)
        df = self.get_occasion_history(estimator, label)
        if self.df is None or not self.df.equals(df):
            self.df = df        
            self.est, self.est_low, self.est_up = rasch_estimate(df, dataset_size)
        horizon = self.est + pos_count
        horizon_low = self.est_low + pos_count
        horizon_up = self.est_up + pos_count
        return horizon, horizon_low, horizon_up

class ParametricRaschHMore(ParametricRaschPython[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    
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
        interaction_cols = {
            f"h{i-1}": len(all_subsets(learner_set, i, i)) 
            for i in range(2, len(all_learners) + 1)
        }
        final_row = {
            **learner_cols,
            **interaction_cols,
            **count_col
        }
        return final_row

class ParametricRaschSingle(ParametricRaschPython[KT, DT, VT, RT, LT], Generic[KT, DT, VT, RT, LT]):
    
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
        interaction_col = {"h1": len(
            all_subsets(
                learner_set, 
                2, len(all_learners) - 1)) 
        }
        final_row = {
            **learner_cols,
            **interaction_col,
            **count_col
        }
        return final_row