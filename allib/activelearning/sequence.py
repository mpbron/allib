# from typing import Any, Optional, Set, Tuple

# import numpy as np

# from .base import AbstractActiveLearner
# from .oracles import OracleFunction

# from ..instances import Instance

# class LeastConfidence(AbstractActiveLearner):
#     def sample(self) -> Optional[Instance]:
#         """Select the document whose  posterior probability is most near 0.5

#         Returns
#         -------
#         Optional[Instance]
#             A tuple containing:
#                 - The document identifier
#                 - The document vector
#                 - The document data
#         """
#         def least_confidence(vector: np.ndarray) -> float:
#             prob_vec = self.estimator.predict_proba(vector)
#             min_prob = np.amin(np.abs(prob_vec - 0.5))
#             return min_prob
#         doc_id = self.argmin(self.unlabeled, least_confidence)
#         if doc_id is not None:
#             return self.unlabeled[doc_id]
#         return None

#     def initialize(self, f_oracle: OracleFunction, it_max: int) -> None:
#         """Label `it_max` randomly selected documents 

#         Parameters
#         ----------
#         f_oracle :
#             Oracle function 
#         it_max : int
#             The number of iterations / documents
#         """
#         for _ in range(it_max):
#             doc_id, vec, data = super().sample()
#             label = f_oracle(doc_id, vec, data, self.labels)
#             self.label_instance(doc_id, label)

#     def iterate_al(self, f_oracle: OracleFunction) -> None:
#         """Perform one iteration of the active learning cycle

#         Parameters
#         ----------
#         f_oracle : [type]
#             Oracle function
#         """
#         # Select cluster and sample data
#         instance = self.sample()

#         # Query the oracle and label the instance
#         label = f_oracle(instance, self.labels)
#         self.label_instance(instance.identifier, label)

#         # Retrain the Machine learning model
#         self.fit()


# class EntropySampling(LeastConfidence):
#     def sample(self) -> Optional[Instance]:
#         """Select the document whose  posterior probability is most near 0.5

#         Returns
#         -------
#         Optional[Instance]
#             A tuple containing:
#                 - The document identifier
#                 - The document vector
#                 - The document data
#         """
#         def entropy(vector: np.ndarray) -> float:
#             prob_vec = self.estimator.predict_proba(vector)
#             entropy = -np.matmul(prob_vec, np.log(prob_vec))
#             return entropy
#         doc_id = self.argmax(self.unlabeled, entropy)
#         if doc_id is not None:
#             return self.unlabeled[doc_id]
#         return None
