import collections
import itertools
import multiprocessing as mp
import warnings
from dataclasses import dataclass
from logging import currentframe
from os import spawnlp
from typing import Any, Deque, Generic, List, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt
import pandas as pd
from numba import (int32, jit, prange, set_num_threads, typed,  # type: ignore
                   types)
from numba.experimental import jitclass
from numba.typed import List as NumbaList
from numpy.linalg import LinAlgError

from ..activelearning.base import ActiveLearner
from ..activelearning.estimator import Estimator
from ..analysis.statistics import EstimationModelStatistics
from ..estimation.base import Estimate
from ..typehints.typevars import DT, KT, LT, RT, VT
from .rasch import EMRaschCombined

spec = [
    ("positive", int32[:]),
    ("estimate", int32[:]),
    ("negative", int32[:]),
    ("observed", int32[:]),
    ("positive_estimate", int32[:]),
    ("negative_estimate", int32[:]),
    ("positive_observed", int32[:]),
    ("negative_observed", int32[:]),
]

@jitclass(spec)
class Masks:
    def __init__(self, 
                 positive: npt.NDArray[Any], 
                 estimate: npt.NDArray[Any], 
                 negative: npt.NDArray[Any], 
                 observed: npt.NDArray[Any], 
                 positive_estimate: npt.NDArray[Any], 
                 negative_estimate: npt.NDArray[Any], 
                 positive_observed: npt.NDArray[Any], 
                 negative_observed: npt.NDArray[Any]):
        self.positive = positive
        self.estimate = estimate
        self.negative = negative
        self.observed = observed
        self.positive_estimate = positive_estimate
        self.negative_estimate = negative_estimate
        self.positive_observed = positive_observed
        self.negative_observed = negative_observed

def create_mask(poses: List[bool], estes: List[bool]) -> Masks:
    def convert(bool_list: Sequence[bool]) -> npt.NDArray[Any]:
        converted = np.array([i for (i,b) in enumerate(bool_list) if b], dtype=np.int32)
        return converted
    positive = convert(poses)
    estimate = convert(estes)
    neges = [not(m) for m in poses]
    obses = [not(m) for m in estes]
    negative = convert(neges)
    observed = convert(obses)
    positive_estimate = convert(NumbaList([est & pos for est, pos in zip(estes, poses)]))
    negative_estimate = convert(NumbaList([est & neg for est, neg in zip(estes, neges)]))
    positive_observed = convert(NumbaList([obs & pos for obs, pos in zip(obses, poses)]))
    negative_observed = convert(NumbaList([obs & neg for obs, neg in zip(obses, neges)]))
    return Masks(positive, estimate, 
                 negative, observed, 
                 positive_estimate, negative_estimate, 
                 positive_observed, negative_observed)

@jit(nopython=True, cache=True) # type: ignore
def l2(b0: npt.NDArray[Any],
       design_mat: npt.NDArray[Any],
       counts: npt.NDArray[Any],
       lam: float = 0.0,
       tolerance: float = 1e-8,
       max_it: int = 10000,
       epsilon: float = 0.1
       ) -> npt.NDArray[Any]:
    counts = counts + epsilon
    current_tolerance = 1
    b = b0
    it = 0
    while current_tolerance > tolerance and it < max_it:
        it = it + 1
        # mu <- exp(D %*% b0)
        mu = np.exp(design_mat @ b0)

        # db <- crossprod(D, Freq) - crossprod(D, mu) - 2 * c(0, lambda * b0[-1])
        all_but_first_b0: npt.NDArray[Any] = b0[1:]
        lbpart: npt.NDArray[Any] = 2 * np.concatenate(
            (
                np.array([0.0]),
                (lam * all_but_first_b0)
        ))
        design_mat_t = design_mat.conj().transpose()
        db = design_mat_t @ counts - design_mat_t @ mu - lbpart

        # I <- crossprod(D, diag(c(mu))) %*% D + 2 * diag(c(0, rep(lambda, length(b0) - 1)))
        repeat_count = b0.shape[0] - 1
        other_lb_part = np.hstack((np.array([0.0]), np.repeat(lam, repeat_count)))
        I = (design_mat_t @ np.diag(mu)) @ design_mat + \
            2 * np.diag(other_lb_part)
        try:
            invI = np.linalg.inv(I)
        except Exception: # LinAlgException
            # Change lambda if the matrix is not invertible
            lam = 1e-6 if lam == 0 else 10 * lam
        else:
            # If inversion succeeded, calculate new coefficients b
            b = b0 + invI @ db
            current_tolerance = np.sum(np.abs(b0 - b))
            b0 = b
    return b


@jit(nopython=True, cache=True)  # type: ignore
def calc_deviance(y: npt.NDArray[Any], y_hat: npt.NDArray[Any]) -> float:
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

@jit(nopython=True, cache=True)  # type: ignore
def parameter_ci(estimate: float, se: float) -> Tuple[float, float]:
    var_min = estimate - 2 * se
    var_max = estimate + 2 * se
    return var_max, var_min

@jit(nopython=True, cache=True)  # type: ignore
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

@jit(nopython=True, cache=True)  # type: ignore
def initial_fit(n_not_read: float, design_mat: npt.NDArray[Any]) -> npt.NDArray[Any]:
    not_read = np.array([n_not_read])
    b0 = np.hstack(
        (
            np.log(not_read),
            np.repeat(0.0, design_mat.shape[1] - 1)
    ))
    return b0

@jit(nopython=True, cache=True)  # type: ignore
def rasch_glm(design_mat: npt.NDArray[Any],
              efit: npt.NDArray[Any],
              counts: npt.NDArray[Any],
              b0: npt.NDArray[Any],
              masks: Masks,
              epsilon: float = 0.0
              ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], float]:
    obs_counts = counts[masks.observed]
    beta = l2(b0, design_mat, efit, epsilon=epsilon)
    mfit = np.exp(design_mat @ beta)
    obs_fitted = mfit[masks.observed]
    deviance = calc_deviance(obs_counts, obs_fitted)
    return beta, mfit, deviance

@jit(nopython=True, cache=True) # type: ignore
def glm(design_mat: npt.NDArray[Any], counts: npt.NDArray[Any], epsilon=0.1) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], float]:
    b0 = np.repeat(1.0, design_mat.shape[1])
    beta = l2(b0, design_mat, counts, epsilon=epsilon)
    mfit = np.exp(design_mat @ beta)
    deviance = calc_deviance(counts, mfit)
    return beta, mfit, deviance

def remove_pos_cells(df: pd.DataFrame) -> pd.DataFrame:
    pos_rows = df[df["positive"] > 0] 
    not_pos_cols = pos_rows.filter(regex="^((?!positive).)*$")
    return not_pos_cols

@jit(nopython=True, parallel=True, cache=True) # type: ignore
def pos_rasch_numpy(design_mat: npt.NDArray[Any], 
                    obs_counts: npt.NDArray[Any], 
                    total_found: int, max_it: int = 2000
                    ) -> Tuple[float, float, float, npt.NDArray[Any], npt.NDArray[Any], float, npt.NDArray[Any]]:
    beta, mfit, deviance = glm(design_mat, obs_counts, epsilon = 0.1)
    estimate = np.exp(beta[0])
    horizon_estimate = int(np.round(total_found + estimate))
    fitted: npt.NDArray[Any] = np.concatenate((np.array([estimate]), mfit))
    p_vals = fitted / np.sum(fitted)
    
    if not np.isnan(estimate):
        deviances = np.zeros(max_it)
        betas = np.zeros((max_it, design_mat.shape[1]))
        mfits = np.zeros((max_it, design_mat.shape[0]))
        multinomial_fits: npt.NDArray[Any] = np.random.multinomial(horizon_estimate, p_vals, max_it)
            
        for idx in prange(max_it):
            multinom = multinomial_fits[idx,1:].copy()
            i_beta, i_mfit, i_deviance = glm(design_mat, multinom, epsilon=0.1)
            betas[idx,:] = i_beta
            mfits[idx,:] = i_mfit
            deviances[idx] = i_deviance
        
        estimates = total_found + np.exp(betas[:,0])
        low = np.percentile(estimates, 2.5)
        up = np.percentile(estimates, 97.5)
        point = np.percentile(estimates, 50)    
        return point, low, up, beta, mfit, deviance, estimates
    return estimate, estimate, estimate, beta, mfit, deviance, np.array([estimate])




def rasch_estimate_only_pos(freq_df: pd.DataFrame, multinomial_size: int = 2000) -> Tuple[Estimate, EstimationModelStatistics]:
    learner_cols = list(freq_df.filter(regex=r"^learner"))
    df_formatted = remove_pos_cells(l2_format(freq_df, learner_cols))   
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





@jit(nopython=True, cache=True)  # type: ignore
def rasch_em(design_mat: npt.NDArray[Any],
             b0: npt.NDArray[Any],
             counts: npt.NDArray[Any],
             n_dataset: float,
             masks: Masks,
             proportion: float,
             tolerance: float = 1e-5,
             max_it=1000) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], float]:
    em_b0 = b0.copy()
    em_counts = counts.copy()
    orig_counts = em_counts[masks.observed]
    n_not_read = n_dataset - np.sum(orig_counts)
    
    n_pos = proportion * n_not_read
    n_neg = (1 - proportion) * n_not_read
    efit = em_counts.copy()

    efit[masks.positive_estimate] = n_pos
    efit[masks.negative_estimate] = n_neg

    beta, mfit, deviance = rasch_glm(design_mat, efit, counts, em_b0, masks)
    current_tolerance = 1
    # Expectation Maximization
    it = 0
    while current_tolerance > tolerance:
        old_deviance = deviance
        efit[masks.estimate] = mfit[masks.estimate] * n_not_read / np.sum(mfit[masks.estimate])
        beta, mfit, deviance = rasch_glm(design_mat, efit, counts, beta, masks)
        current_tolerance = old_deviance - deviance
        if it > max_it or np.any(np.isnan(mfit)):
            #warnings.warn(
            #    f"Exceeded maximum EM iterations ({max_it}). Estimate might not be accurate")
            return beta, mfit, deviance
    return beta, mfit, deviance

@jit(nopython=True, cache=True)  # type: ignore
def rasch_bf(design_mat: npt.NDArray[Any],
             b0: npt.NDArray[Any],
             counts: npt.NDArray[Any],
             n_dataset: float,
             masks: Masks) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    em_b0 = b0.copy()
    em_counts = counts.copy()
    orig_counts = em_counts[masks.observed]
    n_not_read = n_dataset - np.sum(orig_counts)
    
    ar_size = int(n_not_read) + 1

    betas = np.zeros((ar_size, design_mat.shape[1]))
    mfits = np.zeros((ar_size, design_mat.shape[0]))
    deviances = np.zeros((ar_size))
            
    efit = em_counts.copy()

    for n_pos in prange(0, n_not_read + 1):
        n_neg = n_not_read - n_pos
        efit[masks.positive_estimate] = n_pos
        efit[masks.negative_estimate] = n_neg
        
        i_beta, i_mfit, i_deviance = rasch_glm(design_mat, efit, counts, em_b0, masks)
        betas[n_pos,:] = i_beta
        mfits[n_pos,:] = i_mfit
        deviances[n_pos] = i_deviance
    return betas, mfits, deviances

@jit(nopython=True, cache=True)
def select_best_model(betas: npt.NDArray[Any], mfits: npt.NDArray[Any], deviances: npt.NDArray[Any]) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], float]:
    best_beta = np.zeros((betas.shape[1],))
    best_mfit = np.zeros((mfits.shape[1],))
    best_deviance = 2000000.0
    for idx in range(betas.shape[0]):
        current_beta = betas[idx,:]
        current_mfit = mfits[idx,:]
        current_deviance = deviances[idx]
        if current_deviance < best_deviance:
            best_deviance = current_deviance
            best_beta = current_beta
            best_mfit = current_mfit
    return best_beta, best_mfit, best_deviance

@jit(nopython=True, parallel=True, cache=True) # type: ignore
def rasch_numpy(design_mat: npt.NDArray[Any],
                counts: npt.NDArray[Any],
                n_dataset: float,
                masks: Masks,
                proportions: npt.NDArray[Any] = np.array([0.001, 0.01, 0.1, 0.5, 0.75, 0.9]),
                tolerance: float = 1e-5,
                max_it=1000
                ) -> Tuple[npt.NDArray[Any], npt.NDArray[Any], float, float]:
    deviances = np.zeros(proportions.shape)
    betas = np.zeros((proportions.shape[0], design_mat.shape[1]))
    mfits = np.zeros((proportions.shape[0], design_mat.shape[0]))
    num_proportions = proportions.shape[0]
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    b0 = initial_fit(n_not_read, design_mat)
    for idx in prange(num_proportions):
        proportion = proportions[idx]
        beta, mfit, deviance = rasch_em(design_mat, b0,  
                                        counts, n_dataset, masks, 
                                        proportion, tolerance, max_it)
        betas[idx,:] = beta
        mfits[idx,:] = mfit
        deviances[idx] = deviance
    best_beta = np.zeros((design_mat.shape[1],))
    best_mfit = np.zeros((design_mat.shape[0],))
    best_deviance = 2000000.0
    best_proportion = 1.0
    for idx in range(num_proportions):
        current_beta = betas[idx,:]
        current_mfit = mfits[idx,:]
        current_deviance = deviances[idx]
        if current_deviance < best_deviance:
            best_deviance = current_deviance
            best_beta = current_beta
            best_mfit = current_mfit
            best_proportion = proportions[idx]
    return best_beta, best_mfit, best_deviance, best_proportion

@jit(nopython=True, parallel=True, nogil=True, cache=True) # type: ignore
def rasch_parallel(design_mat: npt.NDArray[Any],
                   b0: npt.NDArray[Any], 
                   rounded_mfits: npt.NDArray[Any], 
                   n_dataset: int,
                   proportion: float,
                   masks: Masks):
    # Placeholders for results
    deviances = np.zeros(rounded_mfits.shape[0])
    betas = np.zeros((rounded_mfits.shape[0], design_mat.shape[1]))
    mfits = np.zeros((rounded_mfits.shape[0], design_mat.shape[0]))
    num_mfits = rounded_mfits.shape[0]

    # Parallel execution of Parametric Bootstrap
    for idx in prange(num_mfits):
        mfit_sample = rounded_mfits[idx,:].copy()
        b0_copy = b0.copy()
        beta, mfit, deviance = rasch_em(
            design_mat, b0_copy, mfit_sample,
            n_dataset, masks, proportion) 
        betas[idx,:] = beta
        mfits[idx,:] = mfit
        deviances[idx] = deviance
    
    # Reordering the results
    results = NumbaList()
    for idx in range(num_mfits):
        result = (betas[idx,:], mfits[idx,:], deviances[idx])
        results.append(result)
    return results
        
def rasch_estimate_parametric_approx(freq_df: pd.DataFrame,
                   n_dataset: int,
                   tolerance: float = 1e-5,
                   max_it: int = 2000,
                   multinomial_size: int = 2000) -> Tuple[float, float, float]:
    # Calculate general frequency statistics
    counts: npt.NDArray[Any] = freq_df["count"].values  # type: ignore
    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))

    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
                        .filter(regex=r"^(?:(?!positive$).)+$")
                        .filter(regex=r"^learner")
                        )
    df_formatted = l2_format(df_w_missing, learner_cols)

    # Calculate row masks to select data from the matrices and vectors
    est_mask: List[bool] = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask: List[bool] = list(df_formatted.positive > 0)
    masks = create_mask(pos_mask, est_mask)
    
    design_mat: npt.NDArray[Any] = (
        df_formatted.loc[:, df_formatted.columns != 'count']  # type: ignore
        .values)  # type: ignore

    counts: npt.NDArray[Any] = df_formatted["count"].values  # type: ignore

    only_pos_estimate = rasch_estimate_only_pos(freq_df)[0].point
    proportion: float = only_pos_estimate / (n_dataset - np.sum(counts[masks.observed]))
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    b0 = initial_fit(n_not_read, design_mat)
    beta, mfit, deviance = rasch_em(design_mat, b0, counts, n_dataset, masks, proportion, tolerance, max_it)
    positive_estimate: float = mfit[masks.positive_estimate][0]
    
    # Calculate standard error on predictors
    
    mat_w = np.diag(mfit)
    design_mat_t = design_mat.conj().transpose()
    try:
        vcov = np.linalg.inv(design_mat_t @ mat_w @ design_mat)
        predictors_sampled: npt.NDArray[Any] = np.random.multivariate_normal(
            beta, vcov, size=multinomial_size)
    except LinAlgError:
        warnings.warn("Could not determine confidence interval")
        return positive_estimate, positive_estimate, positive_estimate
    
    sampled_mfits = np.exp(design_mat @ predictors_sampled.T).T
    rounded_mfits = np.round(sampled_mfits)
    low_estimate = positive_estimate
    high_estimate = positive_estimate
    obs_pos = np.sum(counts[masks.positive_observed])
    set_num_threads(7)
    if True:
        results = rasch_parallel(design_mat, beta,  rounded_mfits, n_dataset, proportion, masks)
        estimates: List[float] = [fitted[masks.positive_estimate][0]
                                for (_, fitted, _) in results]
        low_estimate = np.percentile(estimates, 2.5)
        median_estimate = np.percentile(estimates, 50)
        high_estimate = np.percentile(estimates, 97.5)
    return median_estimate, low_estimate, high_estimate

def rasch_estimate_parametric_proportion(freq_df: pd.DataFrame,
                   n_dataset: int,
                   proportion: float,
                   tolerance: float = 1e-5,
                   max_it: int = 2000,
                   multinomial_size: int = 2000) -> Tuple[Estimate, EstimationModelStatistics]:
    # Calculate general frequency statistics
    counts: npt.NDArray[Any] = freq_df["count"].values  # type: ignore
    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))

    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
                        .filter(regex=r"^(?:(?!positive$).)+$")
                        .filter(regex=r"^learner")
                        )
    df_formatted = l2_format(df_w_missing, learner_cols)

    # Calculate row masks to select data from the matrices and vectors
    est_mask: List[bool] = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask: List[bool] = list(df_formatted.positive > 0)
    masks = create_mask(pos_mask, est_mask)
    
    design_mat: npt.NDArray[Any] = (
        df_formatted.loc[:, df_formatted.columns != 'count']  # type: ignore
        .values)  # type: ignore

    counts: npt.NDArray[Any] = df_formatted["count"].values  # type: ignore
    obs_pos = np.sum(counts[masks.positive_observed])
    #only_pos_estimate = rasch_estimate_only_pos(freq_df)[0].point
    
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    b0 = initial_fit(n_not_read, design_mat)
    beta, mfit, deviance = rasch_em(design_mat, b0, counts, n_dataset, masks, proportion, tolerance, max_it)
    positive_estimate: float = mfit[masks.positive_estimate][0]
    horizon_estimate = obs_pos + positive_estimate
    if np.any(np.isnan(mfit)):
        estimate = Estimate(horizon_estimate, horizon_estimate, horizon_estimate)
        stats = EstimationModelStatistics(beta, mfit, deviance, np.array([]))
    else:
        p_vals = mfit / np.sum(mfit)
        multinomial_fits: npt.NDArray[Any] = np.random.multinomial(n_dataset, p_vals, multinomial_size)
        results = rasch_parallel(design_mat, b0, multinomial_fits, n_dataset, proportion, masks)
        estimates: List[float] = [fitted[masks.positive_estimate][0]
                                    for (_, fitted, _) in results]
        stats = EstimationModelStatistics(beta, mfit, deviance, (obs_pos + np.array(estimates)))
        low_estimate = obs_pos + np.percentile(estimates, 2.5)
        high_estimate = obs_pos + np.percentile(estimates, 97.5)
        estimate = Estimate(horizon_estimate, low_estimate, high_estimate)
    return estimate, stats

def rasch_estimate_parametric(freq_df: pd.DataFrame,
                   n_dataset: int,
                   tolerance: float = 1e-5,
                   max_it: int = 2000,
                   multinomial_size: int = 2000) -> Tuple[Estimate, EstimationModelStatistics]:
    # Calculate general frequency statistics
    counts: npt.NDArray[Any] = freq_df["count"].values  # type: ignore
    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))

    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
                        .filter(regex=r"^(?:(?!positive$).)+$")
                        .filter(regex=r"^learner")
                        )
    df_formatted = l2_format(df_w_missing, learner_cols)

    # Calculate row masks to select data from the matrices and vectors
    est_mask: List[bool] = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask: List[bool] = list(df_formatted.positive > 0)
    masks = create_mask(pos_mask, est_mask)
    
    design_mat: npt.NDArray[Any] = (
        df_formatted.loc[:, df_formatted.columns != 'count']  # type: ignore
        .values)  # type: ignore

    counts: npt.NDArray[Any] = df_formatted["count"].values  # type: ignore
    obs_pos = np.sum(counts[masks.positive_observed])
    #only_pos_estimate = rasch_estimate_only_pos(freq_df)[0].point
    
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    proportion: float = 100 / n_not_read
    b0 = initial_fit(n_not_read, design_mat)
    beta, mfit, deviance = rasch_em(design_mat, b0, counts, n_dataset, masks, proportion, tolerance, max_it)
    positive_estimate: float = mfit[masks.positive_estimate][0]
    horizon_estimate = obs_pos + positive_estimate
    if np.any(np.isnan(mfit)):
        estimate = Estimate(horizon_estimate, horizon_estimate, horizon_estimate)
        stats = EstimationModelStatistics(beta, mfit, deviance, np.array([]))
    else:
        p_vals = mfit / np.sum(mfit)
        multinomial_fits: npt.NDArray[Any] = np.random.multinomial(n_dataset, p_vals, multinomial_size)
        rounded_beta = np.around(beta, 3)
        results = rasch_parallel(design_mat, b0, multinomial_fits, n_dataset, proportion, masks)
        estimates: List[float] = [fitted[masks.positive_estimate][0]
                                    for (_, fitted, _) in results]
        stats = EstimationModelStatistics(beta, mfit, deviance, (obs_pos + np.array(estimates)))
        low_estimate = obs_pos + np.percentile(estimates, 2.5)
        high_estimate = obs_pos + np.percentile(estimates, 97.5)
        estimate = Estimate(horizon_estimate, low_estimate, high_estimate)
    return estimate, stats

def rasch_estimate_parametric_no_fixed_proportion(freq_df: pd.DataFrame,
                   n_dataset: int,
                   tolerance: float = 1e-5,
                   max_it: int = 2000,
                   multinomial_size: int = 2000) -> Tuple[Estimate, EstimationModelStatistics]:
    # Calculate general frequency statistics
    counts: npt.NDArray[Any] = freq_df["count"].values  # type: ignore
    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))

    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
                        .filter(regex=r"^(?:(?!positive$).)+$")
                        .filter(regex=r"^learner")
                        )
    df_formatted = l2_format(df_w_missing, learner_cols)

    # Calculate row masks to select data from the matrices and vectors
    est_mask: List[bool] = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask: List[bool] = list(df_formatted.positive > 0)
    masks = create_mask(pos_mask, est_mask)
    
    design_mat: npt.NDArray[Any] = (
        df_formatted.loc[:, df_formatted.columns != 'count']  # type: ignore
        .values)  # type: ignore

    counts: npt.NDArray[Any] = df_formatted["count"].values  # type: ignore
    obs_pos = np.sum(counts[masks.positive_observed])
   
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    b0 = initial_fit(n_not_read, design_mat)
    beta, mfit, deviance, proportion = rasch_numpy(design_mat, counts, n_dataset, masks)
    positive_estimate: float = mfit[masks.positive_estimate][0]
    horizon_estimate = obs_pos + positive_estimate
    if np.any(np.isnan(mfit)):
        estimate = Estimate(horizon_estimate, horizon_estimate, horizon_estimate)
        stats = EstimationModelStatistics(beta, mfit, deviance, np.array([]))
    else:
        p_vals = mfit / np.sum(mfit)
        multinomial_fits: npt.NDArray[Any] = np.random.multinomial(n_dataset, p_vals, multinomial_size)
        results = rasch_parallel(design_mat, b0, multinomial_fits, n_dataset, proportion, masks)
        estimates: List[float] = [fitted[masks.positive_estimate][0]
                                    for (_, fitted, _) in results]
        stats = EstimationModelStatistics(beta, mfit, deviance, (obs_pos + np.array(estimates)))
        low_estimate = obs_pos + np.percentile(estimates, 2.5)
        high_estimate = obs_pos + np.percentile(estimates, 97.5)
        estimate = Estimate(horizon_estimate, low_estimate, high_estimate)
    return estimate, stats

def rasch_estimate_parametric_init_by_pos(freq_df: pd.DataFrame,
                   n_dataset: int,
                   tolerance: float = 1e-5,
                   max_it: int = 2000,
                   multinomial_size: int = 2000) -> Tuple[Estimate, EstimationModelStatistics]:
    # Calculate general frequency statistics
    counts: npt.NDArray[Any] = freq_df["count"].values  # type: ignore
    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))

    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
                        .filter(regex=r"^(?:(?!positive$).)+$")
                        .filter(regex=r"^learner")
                        )
    df_formatted = l2_format(df_w_missing, learner_cols)

    # Calculate row masks to select data from the matrices and vectors
    est_mask: List[bool] = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask: List[bool] = list(df_formatted.positive > 0)
    masks = create_mask(pos_mask, est_mask)
    
    design_mat: npt.NDArray[Any] = (
        df_formatted.loc[:, df_formatted.columns != 'count']  # type: ignore
        .values)  # type: ignore

    counts: npt.NDArray[Any] = df_formatted["count"].values  # type: ignore
    obs_pos = np.sum(counts[masks.positive_observed])
   
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    b0 = initial_fit(n_not_read, design_mat)
    rasch_pos_est, _ = rasch_estimate_only_pos(freq_df, 2000)
    proportion = (rasch_pos_est.point - obs_pos) / n_not_read
    beta, mfit, deviance = rasch_em(design_mat, b0, counts, n_dataset, masks, proportion)
    positive_estimate: float = mfit[masks.positive_estimate][0]
    horizon_estimate = obs_pos + positive_estimate
    if np.any(np.isnan(mfit)):
        estimate = Estimate(horizon_estimate, horizon_estimate, horizon_estimate)
        stats = EstimationModelStatistics(beta, mfit, deviance, np.array([]))
    else:
        p_vals = mfit / np.sum(mfit)
        try:
            multinomial_fits: npt.NDArray[Any] = np.random.multinomial(n_dataset, p_vals, multinomial_size)
        except ValueError:
            estimate = Estimate(horizon_estimate, horizon_estimate, horizon_estimate)
            stats = EstimationModelStatistics(beta, mfit, deviance, np.array([]))
        else:
            results = rasch_parallel(design_mat, b0, multinomial_fits, n_dataset, proportion, masks)
            estimates: List[float] = [fitted[masks.positive_estimate][0]
                                        for (_, fitted, _) in results]
            stats = EstimationModelStatistics(beta, mfit, deviance, (obs_pos + np.array(estimates)))
            low_estimate = obs_pos + np.percentile(estimates, 2.5)
            high_estimate = obs_pos + np.percentile(estimates, 97.5)
            estimate = Estimate(horizon_estimate, low_estimate, high_estimate)
    return estimate, stats

def rasch_estimate(freq_df: pd.DataFrame,
                   n_dataset: int,
                   proportion: float,
                   tolerance: float = 1e-5,
                   max_it: int = 2000) -> Tuple[Estimate, EstimationModelStatistics]:
    # Calculate general frequency statistics
    counts: npt.NDArray[Any] = freq_df["count"].values  # type: ignore
    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))

    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
                        .filter(regex=r"^(?:(?!positive$).)+$")
                        .filter(regex=r"^learner")
                        )
    df_formatted = l2_format(df_w_missing, learner_cols)

    # Calculate row masks to select data from the matrices and vectors
    est_mask: List[bool] = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask: List[bool] = list(df_formatted.positive > 0)
    masks = create_mask(pos_mask, est_mask)

    design_mat: npt.NDArray[Any] = (
        df_formatted.loc[:, df_formatted.columns != 'count']  # type: ignore
        .values)  # type: ignore

    counts: npt.NDArray[Any] = df_formatted["count"].values  # type: ignore
    obs_pos = np.sum(counts[masks.positive_observed])
    #only_pos_estimate = rasch_estimate_only_pos(freq_df)[0].point
    
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    b0 = initial_fit(n_not_read, design_mat)
    beta, mfit, deviance = rasch_em(design_mat, b0, counts, n_dataset, masks, proportion, tolerance, max_it)
    positive_estimate: float = mfit[masks.positive_estimate][0]
    horizon_estimate = obs_pos + positive_estimate
    return Estimate(horizon_estimate, horizon_estimate, horizon_estimate), EstimationModelStatistics(beta, mfit, deviance, np.array([]))

def rasch_estimate_bf(freq_df: pd.DataFrame,
                   n_dataset: int,
                   tolerance: float = 1e-5,
                   max_it: int = 2000) -> Tuple[Estimate, EstimationModelStatistics]:
    # Calculate general frequency statistics
    counts: npt.NDArray[Any] = freq_df["count"].values  # type: ignore
    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))

    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
                        .filter(regex=r"^(?:(?!positive$).)+$")
                        .filter(regex=r"^learner")
                        )
    df_formatted = l2_format(df_w_missing, learner_cols)

    # Calculate row masks to select data from the matrices and vectors
    est_mask: List[bool] = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask: List[bool] = list(df_formatted.positive > 0)
    masks = create_mask(pos_mask, est_mask)

    design_mat: npt.NDArray[Any] = (
        df_formatted.loc[:, df_formatted.columns != 'count']  # type: ignore
        .values)  # type: ignore

    counts: npt.NDArray[Any] = df_formatted["count"].values  # type: ignore
    obs_pos = np.sum(counts[masks.positive_observed])
    #only_pos_estimate = rasch_estimate_only_pos(freq_df)[0].point
    
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    b0 = initial_fit(n_not_read, design_mat)
    beta, mfit, deviance = select_best_model(*rasch_bf(design_mat, b0, counts, n_dataset, masks))
    positive_estimate: float = mfit[masks.positive_estimate][0]
    horizon_estimate = obs_pos + positive_estimate
    return Estimate(horizon_estimate, horizon_estimate, horizon_estimate), EstimationModelStatistics(beta, mfit, deviance, np.array([]))

def rasch_estimate_bf_stats(freq_df: pd.DataFrame,
                   n_dataset: int,
                   proportion: float,
                   tolerance: float = 1e-5,
                   max_it: int = 2000):
    # Calculate general frequency statistics
    counts: npt.NDArray[Any] = freq_df["count"].values  # type: ignore
    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))

    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
                        .filter(regex=r"^(?:(?!positive$).)+$")
                        .filter(regex=r"^learner")
                        )
    df_formatted = l2_format(df_w_missing, learner_cols)

    # Calculate row masks to select data from the matrices and vectors
    est_mask: List[bool] = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask: List[bool] = list(df_formatted.positive > 0)
    masks = create_mask(pos_mask, est_mask)

    design_mat: npt.NDArray[Any] = (
        df_formatted.loc[:, df_formatted.columns != 'count']  # type: ignore
        .values)  # type: ignore

    counts: npt.NDArray[Any] = df_formatted["count"].values  # type: ignore
    obs_pos = np.sum(counts[masks.positive_observed])
    #only_pos_estimate = rasch_estimate_only_pos(freq_df)[0].point
    
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    b0 = initial_fit(n_not_read, design_mat)
    betas, mfits, deviances = rasch_bf(design_mat, b0, counts, n_dataset, masks)
    ests = mfits[:,masks.positive_estimate]
    return ests, deviances 

def rasch_estimate_bf_parametric(freq_df: pd.DataFrame,
                   n_dataset: int,
                   tolerance: float = 1e-5,
                   max_it: int = 2000, 
                   multinomial_size: int = 2000) -> Tuple[Estimate, EstimationModelStatistics]:
    # Calculate general frequency statistics
    counts: npt.NDArray[Any] = freq_df["count"].values  # type: ignore
    # Add place holder rows for the counts that we aim to estimate
    df_w_missing = add_missing_count_rows(freq_df, float("nan"), float("nan"))

    # Change the dataframe in the correct format for the L2 algorithm
    learner_cols = list(df_w_missing
                        .filter(regex=r"^(?:(?!positive$).)+$")
                        .filter(regex=r"^learner")
                        )
    df_formatted = l2_format(df_w_missing, learner_cols)

    # Calculate row masks to select data from the matrices and vectors
    est_mask: List[bool] = list(df_formatted[learner_cols].sum(axis=1) <= 0)
    pos_mask: List[bool] = list(df_formatted.positive > 0)
    masks = create_mask(pos_mask, est_mask)

    design_mat: npt.NDArray[Any] = (
        df_formatted.loc[:, df_formatted.columns != 'count']  # type: ignore
        .values)  # type: ignore

    counts: npt.NDArray[Any] = df_formatted["count"].values  # type: ignore
    obs_pos = np.sum(counts[masks.positive_observed])
        
    n_not_read = n_dataset - np.sum(counts[masks.observed])
    b0 = initial_fit(n_not_read, design_mat)
    beta, mfit, deviance = select_best_model(*rasch_bf(design_mat, b0, counts, n_dataset, masks))
    positive_estimate: float = mfit[masks.positive_estimate][0]
    horizon_estimate = obs_pos + positive_estimate
    proportion = positive_estimate / n_not_read
    if np.any(np.isnan(mfit)):
        estimate = Estimate(horizon_estimate, horizon_estimate, horizon_estimate)
        stats = EstimationModelStatistics(beta, mfit, deviance, np.array([]))
    else:
        p_vals = mfit / np.sum(mfit)
        try:
            multinomial_fits: npt.NDArray[Any] = np.random.multinomial(n_dataset, p_vals, multinomial_size)
        except ValueError:
            estimate = Estimate(horizon_estimate, horizon_estimate, horizon_estimate)
            stats = EstimationModelStatistics(beta, mfit, deviance, np.array([]))
            return estimate, stats
        results = rasch_parallel(design_mat, beta, multinomial_fits, n_dataset, proportion, masks)
        estimates: List[float] = [fitted[masks.positive_estimate][0]
                                    for (_, fitted, _) in results]
        stats = EstimationModelStatistics(beta, mfit, deviance, (obs_pos + np.array(estimates)))
        low_estimate = obs_pos + np.percentile(estimates, 2.5)
        high_estimate = obs_pos + np.percentile(estimates, 97.5)
        estimate = Estimate(horizon_estimate, low_estimate, high_estimate)
    return estimate, stats




class EMRaschRidgeParametricConvPython(
        EMRaschCombined[KT, DT, VT, RT, LT],
        Generic[KT, DT, VT, RT, LT]):
    def __init__(self):
        super().__init__()
        self.df: Optional[pd.DataFrame] = None
        self.est = float("nan")
        self.est_low = float("nan")
        self.est_high = float("nan")
        self.dfs = collections.deque()

    def _start_r(self) -> None:
        pass

    def calculate_estimate(self,
                           estimator: Estimator[Any, KT, DT, VT, RT, LT],
                           label: LT) -> Tuple[float, float, float]:
        pos_count = estimator.env.labels.document_count(label)
        dataset_size = len(estimator.env.dataset)
        df = self.get_occasion_history(estimator, label)
        if self.df is None or not self.df.equals(df):
            self.dfs.append(df)
            self.df = df
            self.est, self.est_low, self.est_up = rasch_estimate_parametric_approx(
                df, dataset_size)
        horizon = self.est + pos_count
        horizon_low = self.est_low + pos_count
        horizon_up = self.est_up + pos_count
        return horizon, horizon_low, horizon_up






class FastEMRaschPosNeg(
        EMRaschCombined[KT, DT, VT, RT, LT],
        Generic[KT, DT, VT, RT, LT]):

    estimates: Deque[Estimate]
    model_info: Deque[EstimationModelStatistics]
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

class FastOnlyPos(FastEMRaschPosNeg):
    def calculate_estimate(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT) -> Estimate:
        df = self.get_occasion_history(estimator, label)
        if not self.dfs or not self.dfs[-1].equals(df):
            self.dfs.append(df)
            est, stats = rasch_estimate_only_pos(df, multinomial_size=self.multinomial_size)
            self.estimates.append(est)
            self.model_info.append(stats)
        return self.estimates[-1]

class FastPosAssisted(FastEMRaschPosNeg):
    def calculate_estimate(self, estimator: Estimator[Any, KT, DT, VT, RT, LT], label: LT) -> Estimate:
        dataset_size = len(estimator.env.dataset)
        df = self.get_occasion_history(estimator, label)
        if not self.dfs or not self.dfs[-1].equals(df):
            self.dfs.append(df)
            est, stats = rasch_estimate_parametric_init_by_pos(df, dataset_size, multinomial_size=self.multinomial_size)
            self.estimates.append(est)
            self.model_info.append(stats)
        return self.estimates[-1]
