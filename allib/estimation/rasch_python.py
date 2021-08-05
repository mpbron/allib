from logging import currentframe
from typing import Any, Generic, List, Optional, Sequence, Tuple

import itertools
import warnings
from numpy.linalg import LinAlgError
import pandas as pd

from ..activelearning.estimator import Estimator
import numpy as np
from .rasch import EMRaschCombined

from ..typehints.typevars import KT, DT, VT, RT, LT

def l2(b0: np.ndarray, 
       design_mat: np.ndarray, 
       counts: np.ndarray,
       N: int,
       lam: float = 0, 
       tolerance: float = 1e-8,
       max_it: int = 10000,
       epsilon: float = 0.1
       ) -> np.ndarray:
    counts = counts + epsilon
    current_tolerance = 1
    b = b0
    it = 0
    while current_tolerance > tolerance and it < max_it:
        it = it + 1

        mu = np.exp(design_mat @ b0)
        all_but_first_b0: np.ndarray = b0[1:]
        lbpart: np.ndarray = 2 * np.concatenate(
            [
                np.array([0]), 
                (lam * all_but_first_b0)
            ])
        design_mat_t = design_mat.conj().transpose()
        db = design_mat_t @ counts  - design_mat_t @ mu - lbpart
        repeat_count = b0.shape[0] - 1
        other_lb_part = np.hstack([[0], np.repeat(lam, repeat_count)])
        I = (design_mat_t @ np.diag(mu)) @ design_mat + 2 * np.diag(other_lb_part)
        try:
            invI = np.linalg.inv(I)
        except LinAlgError:
            # Change lambda if the matrix is not invertible
            lam = 1e-6 if lam == 0 else 10 * lam
        else:
            # If inversion succeeded, calculate new coefficients b
            b = b0 + invI @ db
            current_tolerance = np.sum(np.abs(b0 - b))
            b0 = b
    if it >= max_it:
        warnings.warn(f"Exceeded maximum L2 iterations ({max_it}). Estimate might not be accurate")
    return b

def calc_deviance(y: np.ndarray, y_hat: np.ndarray) -> float:
    pos_y_idx = np.where(y > 0)
    pos_y = y[pos_y_idx]
    deviance = (
        2 * np.sum(pos_y * np.log(pos_y / y_hat[pos_y_idx])) -
        2 * np.sum(y - y_hat))
    return deviance

def add_missing_count_rows(freq_df: pd.DataFrame, n_pos: float, n_neg: float) -> pd.DataFrame:
    pos_dict = {
        **{col: 0 for col in freq_df.columns},
        **{"count": n_pos, "positive": 1}
    }
    neg_dict = {
        **{col: 0 for col in freq_df.columns},
        **{"count": n_neg, "positive": 0}
    }
    new_freq_df = (
        freq_df
            .append(pos_dict, ignore_index=True)
            .append(neg_dict, ignore_index=True)
    )   
    return new_freq_df

def l2_format(freq_df: pd.DataFrame, learner_cols: Sequence[str]) -> pd.DataFrame:
    order_cols = ["positive", *learner_cols]
    df = freq_df.sort_values(order_cols)
    df.insert(0, "intercept", 1)
    return df

def parameter_ci(estimate: float, se: float) -> Tuple[float, float]:
    var_min = estimate - 2 * se
    var_max = estimate + 2 * se
    return var_max, var_min

def calculate_estimate(intercept: float, pos: float) -> float:
    return np.exp(intercept + pos)

def calculate_ci_rasch(intercept: float, 
                       positive: float, 
                       intercept_se: float, 
                       positive_se: float) -> Tuple[float, float]:
    intercepts = parameter_ci(intercept, intercept_se)
    positive_par = parameter_ci(positive, positive_se)
    combinations = itertools.product(intercepts, positive_par)
    results = list(itertools.starmap(calculate_estimate, combinations))
    min_result = min(results)
    max_result = max(results)
    return min_result, max_result


def rasch_estimate(freq_df: pd.DataFrame, 
                n_dataset: int,
                proportion: float = 0.1,
                tolerance: float = 1e-5, max_it: int = 1000) -> Tuple[float, float, float]:
    # Calculate general frequency statistics
    n_notread = n_dataset - np.sum(freq_df["count"].values)
    n_pos = proportion * n_notread
    n_neg = (1 - proportion) * n_dataset

    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))
    
    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
        .filter(regex=r"^(?:(?!positive$).)+$")
        .filter(regex=r"^learner")
        )
    df_formatted = l2_format(df_w_missing, learner_cols)
    
    # Gather some column indices that we need below to access data from the matrices
    positive_col_idx = list(df_formatted.columns).index("positive")
    all_pos_cols = list(df_formatted
        .filter(regex=r"^positive$"))

    # Calculate row masks to select data from the matrices and vectors
    obs_mask = list(df_formatted[learner_cols].sum(axis=1) > 0)
    est_mask = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask = list(df_formatted.positive > 0)
    neg_mask = list(df_formatted.positive <= 0)
    neg_est_mask: List[bool] = [est & neg for est, neg in zip(est_mask, neg_mask)]
    pos_est_mask: List[bool] = [est & pos for est, pos in zip(est_mask, pos_mask)]

    design_mat = (
        df_formatted.loc[:, df_formatted.columns != 'count'] # type: ignore
            .values) # type: ignore
    
    
    obs_counts = df_formatted["count"].values
    efit = df_formatted["count"].values
    efit[pos_est_mask] = n_pos # type: ignore
    efit[neg_est_mask] = n_neg # type: ignore
    mfit = efit

    # Calculate initial fit
    b0 = np.hstack(
        [
            np.log([n_dataset]), 
            np.repeat(0,design_mat.shape[1] - 1)
        ])
    beta = l2(b0, design_mat, efit, n_dataset)
    deviance = calc_deviance(
            obs_counts, 
            np.exp(np.matmul(design_mat, beta)))
    
    current_tolerance = 1
    
    # Expectation Maximization 
    it = 0
    while current_tolerance > tolerance:
        old_deviance = deviance
        mfit = np.exp(design_mat @ beta)
        efit[est_mask] = mfit[est_mask] * n_notread / np.sum(mfit[est_mask])
        beta = l2(beta, design_mat, efit, n_dataset)
        deviance = calc_deviance(
            obs_counts, 
            np.exp(design_mat @ beta))
        current_tolerance = old_deviance - deviance
        if it > max_it:
            warnings.warn(f"Exceeded maximum EM iterations ({max_it}). Estimate might not be accurate")
            return float("nan"), float("nan"), float("nan")


    # Calculate final results
    fitted = np.exp(design_mat @ beta)
    positive_estimate: float = fitted[pos_est_mask][0]

    # Calculate standard error on predictors
    mat_w = np.diag(fitted)
    design_mat_t = design_mat.conj().transpose()
    vcov = np.linalg.inv(design_mat_t @ mat_w @ design_mat)
    se = np.sqrt(np.diagonal(vcov))

    # Calculate confidence interval on estimates    
    lower_bound, upper_bound = calculate_ci_rasch(
        beta[0], 
        beta[positive_col_idx], 
        se[0], 
        se[positive_col_idx])
    
    return positive_estimate, lower_bound, upper_bound

class EMRaschRidgePython(
            EMRaschCombined[KT, DT, VT, RT, LT], 
            Generic[KT, DT, VT, RT, LT]):
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

# def rasch_estimate_delta_method(freq_df: pd.DataFrame, 
#                 n_dataset: int,
#                 proportion: float = 0.1,
#                 tolerance: float = 1e-5) -> Tuple[float, float, float]:
#     # Calculate general frequency statistics
#     n_notread = n_dataset - np.sum(freq_df["count"].values)
#     n_pos = proportion * n_notread
#     n_neg = (1 - proportion) * n_dataset

#     # Add place holder rows for the counts that we aim to estimate
#     df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))
    
#     # Change the dataframe in the correct format for the L2 algorithm
#     learner_cols = list(df_w_missing
#         .filter(regex=r"^(?:(?!positive$).)+$")
#         .filter(regex=r"^learner")
#         )
#     df_formatted = l2_format(df_w_missing, learner_cols)
    
#     # Gather some column indices that we need below to access data from the matrices
#     positive_col_idx = list(df_formatted.columns).index("positive")
#     all_pos_cols = list(df_formatted
#         .filter(regex=r"positive"))

#     # Calculate row masks to select data from the matrices and vectors
#     obs_mask = list(df_formatted[learner_cols].sum(axis=1) > 0)
#     est_mask = list(df_formatted[learner_cols].sum(axis=1) <= 0)
#     pos_mask = list(df_formatted.positive > 0)
#     neg_mask = list(df_formatted.positive <= 0)
#     neg_est_mask = [est & neg for est, neg in zip(est_mask, neg_mask)]
#     pos_est_mask = [est & pos for est, pos in zip(est_mask, pos_mask)]

#     design_mat = (
#         df_formatted.loc[:, df_formatted.columns != 'count'] # type: ignore
#             .values) # type: ignore
    
#     # Calculate start values
#     obs_counts = df_formatted["count"].values
#     efit = df_formatted["count"].values
#     efit[pos_est_mask] = n_pos
#     efit[neg_est_mask] = n_neg
#     mfit = efit

#     # Calculate initial fit
#     b0 = np.hstack(
#         [
#             np.log([n_dataset]), 
#             np.repeat(0,design_mat.shape[1] - 1)
#         ])
#     beta = l2(b0, design_mat, efit, n_dataset)
#     deviance = calc_deviance(
#             obs_counts, 
#             np.exp(np.matmul(design_mat, beta)))
    
#     current_tolerance = 1
    
#     # Expectation Maximization 
#     while current_tolerance > tolerance:
#         old_deviance = deviance
#         mfit = np.exp(design_mat @ beta)
#         efit[est_mask] = mfit[est_mask] * n_notread / np.sum(mfit[est_mask])
#         beta = l2(beta, design_mat, efit, n_dataset)
#         deviance = calc_deviance(
#             obs_counts, 
#             np.exp(design_mat @ beta))
#         current_tolerance = old_deviance - deviance


#     # Calculate final results
#     fitted = np.exp(design_mat @ beta)
#     positive_estimate: float = fitted[pos_est_mask][0]

#     # Calculate standard error on predictors using the delta method
#     mat_w = np.diag(fitted)
#     design_mat_t = design_mat.conj().transpose()
#     vcov = np.linalg.inv(design_mat_t @ mat_w @ design_mat)

    
#     design_mat_pos = np.vstack([
#         design_mat[pos_mask], design_mat[pos_mask]
#     ])
#     design_mat_neg = np.vstack([
#         design_mat[neg_mask], design_mat[neg_mask]
#     ])
#     pred_pos = design_mat_pos @ beta
#     pred_neg = design_mat_neg @ beta

#     deriv_pos = np.exp(pred_pos)
#     deriv_neg = np.exp(pred_neg)

#     jacobian_pos = deriv_pos @ design_mat_pos / design_mat_pos.shape[0]
#     jacobian_neg = deriv_neg @ design_mat_neg / design_mat_neg.shape[0]

#     variance_pos = jacobian_pos @ vcov @ jacobian_pos.transpose()
#     variance_neg = jacobian_neg @ vcov @ jacobian_neg.transpose()
    
#     se_pos = np.sqrt(variance_pos)
    

#     # Calculate confidence interval on estimates    
#     lower_bound = positive_estimate - 2 * se_pos
#     upper_bound = positive_estimate + 2 * se_pos
    
#     return positive_estimate, lower_bound, upper_bound