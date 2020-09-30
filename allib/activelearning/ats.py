# import math
# import random
# from abc import ABC, abstractmethod
# from typing import (Any, AnyStr, Callable, Dict, Iterator, List, Optional, Set,
#                     Tuple)

# import numpy as np
# import pandas as pd
# from numpy.linalg import inv
# from numpy.random import multivariate_normal
# from sklearn.base import BaseEstimator
# from sklearn.cluster import KMeans, MiniBatchKMeans
# from sklearn.metrics.pairwise import cosine_similarity
# from scipy import spatial

# from .base import AbstractActiveLearner
# from .random import RandomSampling
# from .datapoint import ClusteredDataPoint
# from .oracles import OracleFunction

# from ..instances import Instance, InstanceProvider


# class ATSCluster(RandomSampling):
#     identifier: int

#     mean_distance: float
#     var_distance: float
#     super_size: int

#     def __init__(
#             self,
#             provider: InstanceProvider,
#             labels: List[str],
#             identifier: int,
#             datapoints: List[ClusteredDataPoint],
#             super_size: int,
#             estimator: BaseEstimator
#     ) -> None:
#         super().__init__(provider, labels, datapoints, estimator)
#         self.id = identifier

#         vec_mat = np.matrix(np.vstack([d.vector for d in datapoints]))
#         self.mean_distance = np.average(cosine_similarity(vec_mat))
#         self.var_distance = np.var(cosine_similarity(vec_mat))
#         self.super_size = super_size

#     def get_context_vector(self) -> np.ndarray:
#         """Generates a context vector that describes the cluster

#         Returns
#         -------
#         np.ndarray
#             A (normalized) feature vector of size `4 + len(labels)`
#         """
#         vec = [
#             self.mean_distance,
#             self.var_distance,
#             self.size / self.super_size,  # TODO Check if it is better to not scale the size
#             self.get_labeled_ratio()
#         ]
#         for label in self.labels:
#             ratio = len(self.label_dict[label]) / self.size
#             vec.append(ratio)
#         npvec = np.array(vec)
#         return npvec / np.linalg.norm(npvec)


# class ActiveThompsonSampling(AbstractActiveLearner):
#     number_of_clusters: int
#     labels: List[str]

#     cluster_dict: Dict[int, ATSCluster]
#     cluster_dict_doc: Dict[str, ATSCluster]
#     dims: int
#     mat_b: np.ndarray
#     vec_mu_hat: np.ndarray
#     vec_f: np.ndarray
#     v_squared: float = 0.25
#     t: int

#     rewards: List[float]

#     alpha = 0.61
#     beta = 0.12
#     stored_hypothesis: np.ndarray

#     def __init__(
#             self,
#             provider: InstanceProvider,
#             labels: List[str],
#             datapoints: List[ClusteredDataPoint],
#             estimator: BaseEstimator) -> None:
#         super().__init__(provider, labels, datapoints, estimator)

#         cluster_ids = {d.cluster for d in datapoints}
#         cluster_dict = {i: [] for i in cluster_ids}

#         for d in datapoints:
#             cluster_dict[d.cluster].append(d)
#         self.cluster_dict = {
#             i: ATSCluster(provider, labels, i, datapoints, len(datapoints), estimator)
#             for i, datapoints in cluster_dict.items()
#         }
#         self.cluster_dict_doc = {
#             d.index: self.cluster_dict[d.cluster] for d in datapoints}

#         self.t = 0
#         self.dims = 4 + len(labels)

#         self.mat_b = np.identity(self.dims)
#         self.vec_mu_hat = np.zeros(shape=(self.dims))
#         self.vec_f = np.zeros(shape=(self.dims))
#         self.rewards = []
#         self.stored_hypothesis = None

#     def label_instance(self, doc_id: str, label: str) -> None:
#         """Labels the instance and discard it from the unlabeled set

#         Parameters
#         ----------
#         doc_id : str
#             The identifier of the document
#         label : str
#             The label of the document
#         """
#         self.label_dict[label].append(doc_id)
#         self.label_dict_inv[doc_id].append(label)
#         self.unlabeled.discard(doc_id)
#         self.labeled.add(doc_id)

#         # Update the cluster to which the document belongs
#         cluster = self.cluster_dict_doc[doc_id]
#         cluster.label_instance(doc_id, label)

#     def calculate_hypothesis(self) -> np.ndarray:
#         """Calculate the hypothesis, the result of the prediction of all documents in all clusters (flattened)

#         Returns
#         -------
#         np.ndarray
#             The resulting flattened vector
#         """
#         vectors = np.array([vec for vec in self.vector_generator()])
#         pred = self.estimator.predict_proba(vectors)
#         return pred.toarray().flatten()

#     def store_hypothesis(self, hypothesis: np.ndarray) -> None:
#         """Store the hypothesis (overwrite)

#         Parameters
#         ----------
#         hypothesis : np.ndarray
#             The (new) hypothesis
#         """
#         self.stored_hypothesis = hypothesis

#     def decreasing_function(self) -> float:
#         """The decreasing function that scales the rewards 
#            TODO: Check if this is useful or harmful        
#         Returns
#         -------
#         float
#             Decreasing function
#         """
#         return self.alpha * math.exp(-self.beta * self.t)

#     def calculate_reward(self, old_hypothesis, new_hypothesis) -> float:
#         """Calculate the reward, which is the amount in which the old hypothesis differs from the old

#         Parameters
#         ----------
#         old_hypothesis : [type]
#             Old hypothesis
#         new_hypothesis : [type]
#             New hypothesis

#         Returns
#         -------
#         float
#             Reward
#         """
#         cos_sim = 1.0
#         if old_hypothesis is not None:
#             cos_sim = 1 - \
#                 spatial.distance.cosine(old_hypothesis, new_hypothesis)
#         # $y_t = 2 \cdot \frac{\cos^-1(d(h, h_t-1))}{\pi}$
#         yt = 2 * math.acos(cos_sim) / math.pi
#         rt = yt  # / self.decreasing_function()
#         return rt

#     def select_random_cluster(self, non_empty=False) -> ATSCluster:
#         """Select a random cluster (for example, during initialization)

#         Parameters
#         ----------
#         non_empty : bool, optional
#             If we only desire to select clusters that are not empty, by default False

#         Returns
#         -------
#         ATSCluster
#             A randomly selected cluster
#         """
#         if non_empty:
#             cluster_ids = [cluster_id for cluster_id,
#                            c in self.cluster_dict.items() if c.len_unlabeled() > 0]
#         else:
#             cluster_ids = [cluster_id for cluster_id in self.cluster_dict]
#         random_id = random.choice(cluster_ids)
#         return self.cluster_dict[random_id]

#     def select_cluster(self) -> ATSCluster:
#         """Select most optimal Cluster, according to the Active Thompson Sampling algorithm

#         Returns
#         -------
#         ATSCluster
#             The estimated most optimal cluster 
#         """
#         vec_mu_sample = np.random.multivariate_normal(
#             self.vec_mu_hat, self.v_squared * inv(self.mat_b))
#         con_vecs = [(i, c.get_context_vector())
#                     for i, c in self.cluster_dict.items() if c.len_unlabeled() > 0]
#         con_vecs = [(i, np.matmul(np.transpose(c), vec_mu_sample))
#                     for i, c in con_vecs]
#         cluster_id, _ = max(con_vecs, key=lambda v: v[1])
#         return self.cluster_dict[cluster_id]

#     def iterate_al(self, f_oracle: OracleFunction) -> None:
#         """Perform one iteration of the active learning cycle

#         Parameters
#         ----------
#         f_oracle : OracleFunction
#             Oracle function
#         """
#         # Select cluster and sample data
#         cluster = self.select_cluster()
#         context_vector = cluster.get_context_vector()
#         doc_id, vec, data = cluster.sample()

#         # Query the oracle and label the instance
#         label = f_oracle(doc_id, vec, data, self.labels)
#         self.label_instance(doc_id, label)

#         # Calculate the current hypothesis or retrieve it
#         if self.stored_hypothesis is None:
#             old_hypothesis = self.calculate_hypothesis()
#         else:
#             old_hypothesis = self.stored_hypothesis

#         # Retrain the Machine Learning Model
#         self.fit()

#         # Calculate changes in the predictions and calculate the reward
#         new_hypothesis = self.calculate_hypothesis()
#         reward = self.calculate_reward(old_hypothesis, new_hypothesis)

#         # Store the hypothesis and reward
#         self.rewards.append(reward)
#         self.store_hypothesis(new_hypothesis)

#         # Update sample parameters
#         update_b = context_vector * context_vector.reshape(-1, 1)
#         self.mat_b = self.mat_b + update_b
#         self.vec_f = self.vec_f + context_vector * reward
#         self.vec_mu_hat = np.matmul(inv(self.mat_b), self.vec_f)
#         self.t = self.t+1

#     def initialize(self, f_oracle: OracleFunction, it_max) -> None:
#         """Label `it_max` randomly selected documents 

#         Parameters
#         ----------
#         f_oracle :
#             Oracle function 
#         it_max : 
#             The number of iterations / documents
#         """
#         for _ in range(it_max):
#             doc_id, vec, data = self.sample()
#             label = f_oracle(doc_id, vec, data, self.labels)
#             self.label_instance(doc_id, label)

#     def initial_fit(self) -> None:
#         """Store initial hypothesis
#         """
#         self.fit()
#         self.stored_hypothesis = self.calculate_hypothesis()
