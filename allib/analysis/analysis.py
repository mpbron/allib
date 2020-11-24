# Copyright 2019 The ASReview Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import itertools
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, FrozenSet, Generic, Sequence, TypeVar

import numpy as np  # type: ignore
from pampy import match, match_value
from scipy import stats  # type: ignore

from ..activelearning import ActiveLearner
from ..history import BaseLogger, MemoryLogger
from ..instances.base import Instance
from ..labels.base import LabelProvider
from ..labels.memory import MemoryLabelProvider
# from .statistics import (_find_inclusions, _get_labeled_order,
#                          _get_last_proba_order, _get_limits)

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")

class ResultUnit(Enum):
    PERCENTAGE = "Percentage"
    ABSOLUTE = "Absolute"
    FRACTION = "Fraction"

@dataclass
class BinaryPerformance(Generic[KT]):
    true_positives: FrozenSet[KT]
    true_negatives: FrozenSet[KT]
    false_positives: FrozenSet[KT]
    false_negatives: FrozenSet[KT]

    @property
    def recall(self) -> float:
        tp = len(self.true_positives)
        fn = len(self.false_negatives)
        recall = tp / ( tp + fn)
        return recall

    @property
    def precision(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        precision = tp / (tp + fp)
        return precision
    
    @property
    def accuracy(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        tn = len(self.true_negatives)
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        return accuracy

    @property
    def wss(self) -> float:
        tp = len(self.true_positives)
        fp = len(self.false_positives)
        fn = len(self.false_negatives)
        tn = len(self.true_negatives)
        n = tp + fp + fn + tn
        wss = ((tn + fn) / n) - (1 - (tp / ( tp + fn)))
        return wss

    @property
    def f1(self) -> float:
        return self.f_beta(beta=1)

    def f_beta(self, beta: int=1) -> float:
        b2 = beta*beta
        fbeta = (1 + b2) * (
            (self.precision * self.recall) /
            ((b2 * self.precision) + self.recall)) 
        return fbeta

def label_metrics(truth: LabelProvider[KT, LT], 
                  prediction: LabelProvider[KT, LT], 
                  keys: Sequence[KT], label: LT):
    included_keys = frozenset(keys)
    ground_truth_pos = truth.get_instances_by_label(label).intersection(included_keys)
    pred_pos = prediction.get_instances_by_label(label)
    true_pos = pred_pos.intersection(ground_truth_pos)
    false_pos = pred_pos.difference(true_pos)
    false_neg = ground_truth_pos.difference(true_pos)
    true_neg = included_keys.difference(true_pos, false_pos, false_neg)
    return BinaryPerformance[KT](true_pos, true_neg, false_pos, false_neg)

def classifier_performance(learner: ActiveLearner[KT, DT, VT, RT, LT], 
              ground_truth: LabelProvider[KT, LT],
              instances: Sequence[Instance[KT, DT, VT, RT]]) -> Dict[LT, BinaryPerformance[KT]]:
    keys = [ins.identifier for ins in instances]
    labelset = learner.env.labels.labelset
    pred_provider = MemoryLabelProvider[KT, LT](labelset, {}, {})
    predictions = learner.predict(instances)
    for ins, pred in zip(instances, predictions):
        pred_provider.set_labels(ins, *pred)
    performance = {
        label: label_metrics(
            ground_truth, 
            pred_provider, 
            keys,
            label
        ) for label in labelset
    }
    return performance

def process_performance(learner: ActiveLearner[KT, DT, VT, RT, LT], label: LT) -> BinaryPerformance[KT]:
    labeled = frozenset(learner.env.labeled)
    labeled_positives = learner.env.labels.get_instances_by_label(label)
    labeled_negatives = labeled.difference(labeled_positives)
    
    truth_positives = learner.env.truth.get_instances_by_label(label)
   
    unlabeled  = frozenset(learner.env.unlabeled)
    unlabeled_positives = unlabeled.intersection(learner.env.truth.get_instances_by_label(label))
    unlabeled_negatives = unlabeled.difference(unlabeled_positives)
    
    true_positives = labeled_positives.intersection(truth_positives)
    false_positives = labeled_positives.difference(truth_positives).union(labeled_negatives)
    false_negatives = truth_positives.difference(labeled_positives)
    true_negatives = unlabeled_negatives

    return BinaryPerformance(true_positives, true_negatives, false_positives, false_negatives)

# class Analysis():
#     """Analysis object to do statistical analysis on state files."""
#     def __init__(self, history: MemoryLogger, key: str):
#         # Sometimes an extra dataset is present in the state_file(s).
#         # These signify not the labels on which the model was trained, but the
#         # ones that were included in the end (or some other intermediate step.
#         self.history = history
#         self.final_labels = None
#         self.labelset = self.history.labelset
#         self.empty = True

#         self.key = key
#         self.num_runs = len(self.history.snapshots)
       
#     def inclusions_found(self,
#                          result_format: ResultUnit,
#                          final_labels=False,
#                          **kwargs):
#         """Get the number of inclusions at each point in time.

#         Caching is used to prevent multiple calls being expensive.

#         Arguments
#         ---------
#         result_format: str
#             The format % or # of the returned values.
#         final_labels: bool
#             If true, use the final_labels instead of labels for analysis.

#         Returns
#         -------
#         tuple:
#             Three numpy arrays with x, y, error_bar.
#         """
#         if final_labels:
#             labels = self.final_labels
#         else:
#             labels = self.labels

#         fl = final_labels
#         if fl not in self.inc_found:
#             # Compute the comclusions if not found in cache.
#             self.inc_found[fl] = {}
#             avg, err, iai, ninit = self._get_inc_found(labels=labels, **kwargs)
#             self.inc_found[fl]["avg"] = avg
#             self.inc_found[fl]["err"] = err
#             self.inc_found[fl]["inc_after_init"] = iai
#             self.inc_found[fl]["n_initial"] = ninit
#         dx = 0
#         dy = 0

#         x_norm = len(labels) - self.inc_found[fl]["n_initial"]
#         y_norm = self.inc_found[fl]["inc_after_init"]

#         if result_format == "percentage":
#             x_norm /= 100
#             y_norm /= 100
#         elif result_format == "number":
#             x_norm /= len(labels)
#             y_norm /= self.inc_found[fl]["inc_after_init"]

#         norm_xr = (np.arange(1,
#                              len(self.inc_found[fl]["avg"]) + 1) - dx) / x_norm
#         norm_yr = (np.array(self.inc_found[fl]["avg"]) - dy) / y_norm
#         norm_y_err = np.array(self.inc_found[fl]["err"]) / y_norm

#         return norm_xr, norm_yr, norm_y_err

#     def _get_inc_found(self, labels=False):
#         """Get the number of inclusions (without formatting)."""
#         inclusions_found = []

#         for snapshot in self.history.snapshots:
#             inclusions, inc_after_init, n_initial = _find_inclusions(
#                 state, labels)
#             inclusions_found.append(inclusions)

#         inc_found_avg = []
#         inc_found_err = []
#         for i_instance in itertools.count():
#             cur_vals = []
#             for i_file in range(self.num_runs):
#                 try:
#                     cur_vals.append(inclusions_found[i_file][i_instance])
#                 except IndexError:
#                     pass
#             if len(cur_vals) == 0:
#                 break
#             if len(cur_vals) == 1:
#                 err = cur_vals[0]
#             else:
#                 err = stats.sem(cur_vals)
#             avg = np.mean(cur_vals)
#             inc_found_avg.append(avg)
#             inc_found_err.append(err)

#         if self.num_runs == 1:
#             inc_found_err = np.zeros(len(inc_found_err))

#         return inc_found_avg, inc_found_err, inc_after_init, n_initial

#     def wss(self, val=100, x_format="percentage", **kwargs):
#         """Get the WSS (Work Saved Sampled) value.

#         Arguments
#         ---------
#         val:
#             At which recall, between 0 and 100.
#         x_format:
#             Format for position of WSS value in graph.

#         Returns
#         -------
#         tuple:
#             Tuple consisting of WSS value, x_positions, y_positions of WSS bar.
#         """
#         norm_xr, norm_yr, _ = self.inclusions_found(result_format="percentage",
#                                                     **kwargs)

#         if x_format == "number":
#             x_return, y_result, _ = self.inclusions_found(
#                 result_format="number", **kwargs)
#             y_max = self.inc_found[False]["inc_after_init"]
#             y_coef = y_max / (len(self.labels) -
#                               self.inc_found[False]["n_initial"])
#         else:
#             x_return = norm_xr
#             y_result = norm_yr
#             y_max = 1.0
#             y_coef = 1.0

#         for i in range(len(norm_yr)):
#             if norm_yr[i] >= val - 1e-6:
#                 return (norm_yr[i] - norm_xr[i], (x_return[i], x_return[i]),
#                         (x_return[i] * y_coef, y_result[i]))
#         return (None, None, None)

#     def rrf(self, val=10, x_format="percentage", **kwargs):
#         """Get the RRF (Relevant References Found).

#         Arguments
#         ---------
#         val:
#             At which recall, between 0 and 100.
#         x_format:
#             Format for position of RRF value in graph.

#         Returns
#         -------
#         tuple:
#             Tuple consisting of RRF value, x_positions, y_positions of RRF bar.

#         """
#         norm_xr, norm_yr, _ = self.inclusions_found(result_format="percentage",
#                                                     **kwargs)

#         if x_format == "number":
#             x_return, y_return, _ = self.inclusions_found(
#                 result_format="number", **kwargs)
#         else:
#             x_return = norm_xr
#             y_return = norm_yr

#         for i in range(len(norm_yr)):
#             if norm_xr[i] >= val - 1e-6:
#                 return (norm_yr[i], (x_return[i], x_return[i]), (0,
#                                                                  y_return[i]))
#         return (None, None, None)

#     def avg_time_to_discovery(self, result_unit: ResultUnit = ResultUnit.ABSOLUTE):
#         """Estimate the Time to Discovery (TD) for each paper.

#         Get the best/last estimate on how long it takes to find a paper.

#         Arguments
#         ---------
#         result_format: str
#             Desired output format: "number", "fraction" or "percentage".

#         Returns
#         -------
#         dict:
#             For each inclusion, key=paper_id, value=avg time.
#         """
#         labels = self.labels
#         one_labels = np.where(labels == 1)[0]
#         time_results = {label: [] for label in one_labels}

#         # Iterate over all state files
#         for state in self.states.values():
#             # Get the order in which records were labeled
#             label_order, n = _get_labeled_order(state)
#             # Get the ranking of all papers at the last query
#             proba_order = _get_last_proba_order(state)

#             # Adjust factor, depending on the desired output format
#             time_mult: float = match(result_unit,
#                 ResultUnit.ABSOLUTE, 1.0,
#                 ResultUnit.FRACTION, 1 / (len(labels) - n),
#                 ResultUnit.PERCENTAGE, 100 / (len(labels) - n)
#             ) 
#             # Get the time to discovery
#             for i_time, idx in enumerate(label_order[n:]):
#                 # for all inclusions that were found/labeled
#                 if labels[idx] == 1:
#                     time_results[idx].append(time_mult * (i_time + 1))
#             for i_time, idx in enumerate(proba_order):
#                 # for all inclusions that weren't found/labeled
#                 if labels[idx] == 1 and idx not in label_order[:n]:
#                     time_results[idx].append(time_mult *
#                                              (i_time + len(label_order) + 1))

#         results = {}

#         # Merge the results of all state files
#         for label, trained_time in time_results.items():
#             if len(trained_time) > 0:
#                 results[label] = np.average(trained_time)

#         return results

#     def limits(self, prob_allow_miss=[0.1], result_format="percentage"):
#         """For each query, compute the number of papers for a criterium.

#         A criterium is the average number of papers missed. For example,
#         with 0.1, the criterium is that after reading x papers, there is
#         (about) a 10% chance that one paper is not included. Another example,
#         with 2.0, there are on average 2 papers missed after reading x papers.
#         The value for x is returned for each query and probability by the
#         function.

#         Arguments
#         ---------
#         prob_allow_miss: list, float
#             Sets the criterium for how many papers can be missed.

#         returns
#         -------
#         dict:
#             One entry, "x_range" with the number of papers read.
#             List, "limits" of results for each probability and
#             at # papers read.
#         """
#         if not isinstance(prob_allow_miss, list):
#             prob_allow_miss = [prob_allow_miss]
#         state = self.states[self._first_file]
#         n_queries = state.n_queries()
#         results = {
#             "x_range": [],
#             "limits": [[] for _ in range(len(prob_allow_miss))],
#         }

#         n_train = 0
#         _, n_initial = _get_labeled_order(state)
#         for query_i in range(n_queries):
#             new_limits = _get_limits(self.states,
#                                      query_i,
#                                      self.labels,
#                                      proba_allow_miss=prob_allow_miss)

#             try:
#                 new_train_idx = state.get("train_idx", query_i)
#             except KeyError:
#                 new_train_idx = None

#             if new_train_idx is not None:
#                 n_train = len(new_train_idx)

#             if new_limits is not None:
#                 if result_format == "percentage":
#                     normalizer = 100 / (len(self.labels) - n_initial)
#                 else:
#                     normalizer = 1
#                 results["x_range"].append((n_train - n_initial) * normalizer)
#                 for i_prob in range(len(prob_allow_miss)):
#                     results["limits"][i_prob].append(
#                         (new_limits[i_prob] - n_initial) * normalizer)

#         if result_format == "percentage":
#             res_dtype = np.float
#         else:
#             res_dtype = np.int

#         results["x_range"] = np.array(results["x_range"], dtype=res_dtype)
#         for i_prob in range(len(prob_allow_miss)):
#             results["limits"][i_prob] = np.array(results["limits"][i_prob],
#                                                  res_dtype)
#         return results

#     def close(self):
#         """Close states."""
#         for state in self.states.values():
#             state.close()
