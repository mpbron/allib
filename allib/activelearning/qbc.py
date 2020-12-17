# from __future__ import annotations

# import functools
# from abc import ABC, abstractstaticmethod
# from typing import List, Optional

# import numpy as np
# from sklearn.exceptions import NotFittedError
# from scipy.stats import entropy

# from ..environment import AbstractEnvironment
# from ..instances import Instance
# from ..machinelearning import AbstractClassifier

# from .base import ActiveLearner
# from .ml_based import MLBased
# from .poolbased import PoolbasedAL
# from .random import RandomSampling

# class QBC(PoolbasedAL):
#     _name = "QBC"
#     def __init__(self, classifier_list: List[AbstractClassifier]):
#         super().__init__(None)
#         self.classifiers = classifier_list
#         self.fallback = RandomSampling(classifier_list[0])
#         self.commitee_size = len(classifier_list)

#     def __call__(self, environment: AbstractEnvironment) -> QBC:
#         super().__call__(environment)
#         self.fallback = self.fallback(environment)
#         for classifier in self.classifiers:
#             classifier(environment)
#         return self

#     @staticmethod
#     def selection_criterion(prob_mat: np.ndarray) -> np.ndarray:
#         return - entropy(prob_vec, axis=0)

#     @MLBased.query_fallback
#     @ActiveLearner.query_log
#     def query(self) -> Optional[Instance]:
#         """Select the instance whose posterior probability is most near 0.5

#         Returns
#         -------
#         Optional[Instance]
#             An object containing:
#                 - The document identifier
#                 - The document vector
#                 - The document data
#         """
#         unlabeled_vectors = self._unlabeled.feature_matrix.matrix
#         commitee_votes = []
#         for classifier in self.classifiers:
#             member_votes = classifier.predict(unlabeled_vectors)
#             commitee_votes.append(member_votes)
#         total_votes = np.array(commitee_votes).sum(axis=0) / self.commitee_size
#         metric_result = self.selection_criterion(total_votes)
#         np_idx = np.argmax(metric_result)
#         doc_id = self._unlabeled.feature_matrix.get_instance_id(np_idx)
#         if doc_id is not None:
#             return self._unlabeled[doc_id]
#         return None

#     @MLBased.query_batch_fallback
#     @ActiveLearner.query_batch_log
#     def query_batch(self, batch_size: int) -> List[Instance]:
#         unlabeled_vectors = self._unlabeled.feature_matrix.matrix
#         commitee_votes = []
#         for classifier in self.classifiers:
#             member_votes = classifier.predict(unlabeled_vectors)
#             commitee_votes.append(member_votes)
#         total_votes = np.array(commitee_votes).sum(axis=0) / self.commitee_size
#         metric_result = self.selection_criterion(total_votes)
#         np_idxs = np.flip(np.argsort(metric_result[np.argpartition(metric_result, -batch_size)])).tolist()
#         results = []
#         for np_idx in np_idxs:
#             doc_id = self._unlabeled.feature_matrix.get_instance_id(np_idx)
#             if doc_id is not None:
#                 results.append(self._unlabeled[doc_id])
#         return results

