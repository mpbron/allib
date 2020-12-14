# import json
# import os
# from pathlib import Path

# import numpy as np  # type: ignore
# from matplotlib import pyplot as plt  # type: ignore
# from scipy.optimize import minimize  # type: ignore

# from .exptailnorm import ExpTailNorm
# from .download import optimize_distribution # type: ignore
# from ..activelearning import PoolBasedAL

# from typing import TypeVar, Optional, Generic, Union, AnyStr

# KT = TypeVar("KT")
# VT = TypeVar("VT")
# DT = TypeVar("DT")
# RT = TypeVar("RT")
# LT = TypeVar("LT")


# def error_estimate(learner: PoolBasedAL[KT, DT, np.ndarray, RT, LT], state_fp, data_fp, opt_results):
#     labels = learner.env.labels.labelset
#     inclusion_est = []
#     prob_finished = []
#     cur_included = []
#     perc_reviewed = []
#     return []

# def plot_results(error_data):
#     perc_reviewed = error_data["perc_reviewed"]
#     inclusions_est = error_data["inclusion_est"]
#     prob_finished = error_data["prob_finished"]
#     cur_included = error_data["cur_included"]
#     n_total_inclusions = error_data["n_total_inclusions"]

#     plt.xlabel("% reviewed")
#     plt.ylabel("Number of inclusions")
#     plt.plot(perc_reviewed, inclusions_est, label="estimate")
#     plt.plot(perc_reviewed, cur_included, label="found")
#     plt.legend(loc="lower right")
#     plt.show()

#     plt.xlabel("% reviewed")
#     plt.plot(perc_reviewed, prob_finished, label="Estimate @100%")
#     plt.plot(perc_reviewed, np.array(cur_included)/n_total_inclusions,
#                 label="Fraction of inclusions found")
#     plt.legend(loc="lower right")
#     plt.show()

# def discrete_norm_dist(dist, train_percentage, bins):
#     norm_cdf = dist.cdf(bins)
#     norm_pdf = train_percentage*(norm_cdf[1:]-norm_cdf[:-1])
#     norm_hist = norm_pdf/norm_pdf.sum()
#     return norm_hist/(bins[1]-bins[0])


# def percentage_found(norm_opt_cum_df, train_percentage, bins, mu, sigma):
#     normalized_bins = (bins-mu)/sigma
#     x_cum = norm_opt_cum_df[0]
#     y_cum = norm_opt_cum_df[1]
#     d_cum = y_cum[1]-y_cum[0]

#     prob_found = 0
#     for i_bin in range(len(train_percentage)):
#         bin_start = normalized_bins[i_bin]
#         bin_end = normalized_bins[i_bin+1]

#         i_cum_start = np.searchsorted(x_cum, bin_start)
#         i_cum_end = np.searchsorted(x_cum, bin_end)

#         if i_cum_start == len(x_cum):
#             continue

#         y_start = y_cum[i_cum_start] - d_cum
#         if i_cum_end == len(x_cum):
#             y_end = 1
#         else:
#             y_end = y_cum[i_cum_end] - d_cum

#         prob = y_end-y_start
#         prob_found += prob*train_percentage[i_bin]

#     return prob_found


# def prob_all_found(min_df, df_pool, mu, sigma):
#     df_max = np.max(df_pool)
#     df_max_norm = (df_max-mu)/sigma

#     x_min_df = min_df[0]
#     y_min_df = min_df[1]

#     i_min_df = np.searchsorted(x_min_df, df_max_norm)

#     if i_min_df == 0:
#         return 1.0
#     else:
#         return 1-y_min_df[i_min_df - 1]


# def log_likelihood(train_dist, expected_dist):
#     likelihood = 0
#     for i_bin in range(len(expected_dist)):
#         if train_dist[i_bin]:
#             likelihood += train_dist[i_bin] * np.log(expected_dist[i_bin])
#     return -likelihood


# def corrected_proba(X, y, model, balance_model, train_one_idx, train_zero_idx,
#                     n_sample=10):
#     cor_proba = []
#     for _ in range(n_sample):
#         if len(train_one_idx) == 1:
#             new_train_idx = np.append(train_one_idx, train_zero_idx)
#             X_train, y_train = balance_model.sample(X, y, new_train_idx, {})
#             model.fit(X_train, y_train)
#             correct_proba = model.predict_proba(X[train_one_idx])[0, 1]
#             cor_proba.append(correct_proba)
#             continue

#         for i_rel_train in range(len(train_one_idx)):
#             new_train_idx = np.append(np.delete(train_one_idx, i_rel_train),
#                                       train_zero_idx)
#             X_train, y_train = balance_model.sample(X, y, new_train_idx, {})
#             model.fit(X_train, y_train)
#             correct_proba = model.predict_proba(
#                 X[train_one_idx[i_rel_train]])[0, 1]
#             cor_proba.append(correct_proba)

#     return np.array(cor_proba)


# def estimate_inclusions(train_idx, pool_idx, X, y, opt_results, model,
#                         balance_model):

#     X_train, y_train = balance_model.sample(X, y, train_idx, {})

#     model.fit(X_train, y_train)
#     proba = model.predict_proba(X)[:, 1]
#     df_all_corrected = -np.log(1/proba-1)

#     train_one_idx = train_idx[np.where(y[train_idx] == 1)[0]]
#     train_zero_idx = train_idx[np.where(y[train_idx] == 0)[0]]

#     correct_one_proba = corrected_proba(X, y, model, balance_model,
#                                         train_one_idx,
#                                         train_zero_idx)

#     df_one_corrected = -np.log(1/correct_one_proba-1)
#     df_train_corrected = df_all_corrected[train_idx]
#     df_train_zero = df_all_corrected[train_zero_idx]
#     df_pool = df_all_corrected[pool_idx]

#     df_all = np.concatenate((df_all_corrected, df_one_corrected))
#     h_min = np.min(df_all)
#     h_max = np.max(df_all)
#     h_range = (h_min, h_max)
#     n_bins = 40

#     hist, bin_edges = np.histogram(df_one_corrected, bins=n_bins,
#                                    range=h_range, density=True)
#     hist_all, _ = np.histogram(df_all_corrected, bins=n_bins, range=h_range,
#                                density=False)
#     hist_pool, _ = np.histogram(df_pool, bins=n_bins, range=h_range,
#                                 density=False)
#     hist_train_zero, _ = np.histogram(df_train_zero, bins=n_bins, range=h_range,
#                                       density=False)
#     hist_train_one, _ = np.histogram(df_one_corrected, bins=n_bins, range=h_range,
#                                      density=False)
#     hist_train, _ = np.histogram(df_train_corrected, bins=n_bins,
#                                  range=h_range, density=False)

#     perc_train = (hist_train_zero + hist_train_one/10 + 0.000001)/(
#         hist_train_zero + hist_train_one/10 + hist_pool + 0.000001)

#     def guess_func(x):
#         dist = ExpTailNorm(*x, *opt_results["extra_param"])
#         corrected_dist = discrete_norm_dist(dist, perc_train, bin_edges)
#         return log_likelihood(hist, corrected_dist)

#     mu_range = h_range
#     est_sigma = np.sqrt(np.var(df_one_corrected))
#     sigma_range = (0.7*est_sigma, 1.3*est_sigma)
#     x0 = np.array((np.average(df_one_corrected), est_sigma))

#     minim_result = minimize(fun=guess_func, x0=x0,
#                             bounds=[mu_range, sigma_range])

#     param = minim_result.x
#     est_true_dist = ExpTailNorm(*param, *opt_results["extra_param"])

#     est_found_dist = discrete_norm_dist(est_true_dist, perc_train, bin_edges)

#     perc_found = percentage_found(opt_results["cum_df"], perc_train, bin_edges,
#                                   param[0], param[1])

#     p_all_found = prob_all_found(
#         opt_results["min_df"], df_pool, param[0], param[1])
#     return np.sum(y[train_idx])/perc_found, p_all_found
