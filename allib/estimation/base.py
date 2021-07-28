from abc import ABC, abstractmethod
from allib.activelearning.base import ActiveLearner
from dataclasses import dataclass
import itertools
from collections import Counter
import pandas as pd # type: ignore
from typing import Any, FrozenSet, Generic, Optional, Tuple, TypeVar, List, Sequence, Set
from ..environment import AbstractEnvironment
from ..activelearning.ml_based import MLBased
from ..machinelearning import AbstractClassifier
from ..activelearning.ml_based import FeatureMatrix, MLBased
from sklearn.linear_model import LogisticRegression # type: ignore

import numpy as np # type: ignore

from ..typehints import IT, KT, DT, VT, RT, LT

class AbstractEstimator(ABC, Generic[IT, KT, DT, VT, RT, LT]):
    name = "AbstractEstimator"
    @abstractmethod
    def __call__(self, 
                 learner: ActiveLearner[Any, KT, DT, VT, RT, LT], label: LT
                ) -> Tuple[float, float, float]:
        raise NotImplementedError

@dataclass
class DecisionRow(Generic[KT]):
    key: KT
    probability: float
    labeled: bool
    pos_labeled: bool
    order: Optional[int]

class SemiEstimator(AbstractEstimator[IT, KT, DT, np.ndarray, RT, LT]):
    @abstractmethod
    def semi(self, 
             learner: MLBased[IT, KT, DT, np.ndarray, RT, LT, np.ndarray, np.ndarray], 
             pos_label: LT) -> Tuple[float, Optional[float], Optional[float]]:
        def temporary_label_indices(y_pred_proba: np.ndarray) -> List[int]:
            order = np.argsort(y_pred_proba)[::-1]
            print(f"{y_pred_proba[order[0]]}, {y_pred_proba[order[-1]]}")
            count = 0
            target = 1
            candidates: List[int] = []
            sample: List[int] = []
            for i, x in enumerate(y_pred_proba[order]):
                count = count + x
                candidates.append(order[i])
                if count >= target:
                    sample.append(candidates[0])
                    target = target + 1
                    candidates = []
            return sample
        env = learner.env
        classifier = learner.classifier
        def get_decision_values(pos_labeled: FrozenSet[KT], proba = True):
            pos_col_idx = classifier.get_label_column_index(pos_label)
            for mat in FeatureMatrix[KT].generator_from_provider(env.dataset, 100):
                decision_keys = mat.indices
                if proba:
                    decision_proba: List[float] = classifier.predict_proba(mat.matrix)[:,pos_col_idx].tolist() # type: ignore
                else:
                    decision_proba: List[float] = classifier.innermodel.decision_function(mat.matrix).tolist() # type: ignore
                decision_pos_labeled = map(lambda key: key in pos_labeled, decision_keys)
                order = map(lambda key: env.logger.get_label_order(key) if key in env.labeled else None, decision_keys)
                decision_labeled = map(lambda key: key in env.labeled, decision_keys)
                rows = itertools.starmap(
                    DecisionRow[KT], zip(
                        decision_keys, 
                        decision_proba, 
                        decision_labeled, 
                        decision_pos_labeled, 
                        order)
                    )
                yield from rows

        pos_labeled = env.labels.get_instances_by_label(pos_label)
        neg_labeled = frozenset(env.labeled).difference(pos_labeled)
        pos_label_count = len(pos_labeled)
        neg_label_count = len(neg_labeled)

        decision_df = pd.DataFrame(list(get_decision_values(pos_labeled)))

        # all_idx = decision_df.index.values
        pos_idx: List[int] = decision_df.index[decision_df.pos_labeled] # type: ignore
        # neg_idx = decision_df.index[not decision_df.pos_label]
        unl_idx = decision_df.index[decision_df.labeled == False]
        x_data = decision_df["probability"].values.reshape(-1, 1) 
        y_data = decision_df["pos_labeled"].values.astype(np.int64)

        def estimate(x_data, y_data, unl_idx, neg_label_count, pos_num_last):
            regularization_strength = sum(y_data) / neg_label_count 
            lreg = LogisticRegression(
                penalty='l2', 
                fit_intercept=True,
                balanced=True,
                C=regularization_strength)
            lreg.fit(x_data, y_data)
            pos_at = list(lreg.classes_).index(1)
            x_unl = x_data[unl_idx]
            y_pred_proba = lreg.predict_proba(x_unl)[:, pos_at]
            idx_temp_positive = temporary_label_indices(y_pred_proba)
            y_data_new = np.copy(y_data)
            for idx in unl_idx[idx_temp_positive]:
                y_data_new[idx] = 1
            pos_num = sum(y_data_new)
            if pos_num == pos_num_last:
                return pos_num        
            return estimate(x_data, y_data_new, unl_idx, neg_label_count, pos_num)
        r_estimate = estimate(x_data, y_data, unl_idx, neg_label_count, 0)
        return r_estimate, 0.0, 0.0



def semi(learner: MLBased[IT, Any, Any, np.ndarray, Any, LT, np.ndarray, np.ndarray], pos_label: LT, neg_label: LT) -> int:
    def temporary_label_indices(y_pred_proba: np.ndarray) -> List[int]:
        order = np.argsort(y_pred_proba)[::-1]
        print(f"{y_pred_proba[order[0]]}, {y_pred_proba[order[-1]]}")
        count = 0
        target = 1
        candidates: List[int] = []
        sample: List[int] = []
        for i, x in enumerate(y_pred_proba[order]):
            count = count + x
            candidates.append(order[i])
            if count >= target:
                sample.append(candidates[0])
                target = target + 1
                candidates = []
        return sample
    env = learner.env
    classifier = learner.classifier
    def get_decision_values(pos_labeled: FrozenSet[KT], proba = True):
        pos_col_idx = classifier.get_label_column_index(pos_label)
        for mat in FeatureMatrix[KT].generator_from_provider(env.dataset, 100):
            decision_keys = mat.indices
            if proba:
                decision_proba: List[float] = classifier.predict_proba(mat.matrix)[:,pos_col_idx].tolist() # type: ignore
            else:
                decision_proba: List[float] = classifier.innermodel.decision_function(mat.matrix).tolist() # type: ignore
            decision_pos_labeled = map(lambda key: key in pos_labeled, decision_keys)
            order = map(lambda key: env.logger.get_label_order(key) if key in env.labeled else None, decision_keys)
            decision_labeled = map(lambda key: key in env.labeled, decision_keys)
            rows = itertools.starmap(
                DecisionRow[KT], zip(
                    decision_keys, 
                    decision_proba, 
                    decision_labeled, 
                    decision_pos_labeled, 
                    order)
                )
            yield from rows

    pos_labeled = env.labels.get_instances_by_label(pos_label)
    neg_labeled = env.labels.get_instances_by_label(neg_label)
    pos_label_count = env.labels.document_count(pos_label)
    neg_label_count = env.labels.document_count(neg_label)
    
    decision_df = pd.DataFrame(list(get_decision_values(pos_labeled)))
    
    # all_idx = decision_df.index.values
    pos_idx: List[int] = decision_df.index[decision_df.pos_labeled] # type: ignore
    # neg_idx = decision_df.index[not decision_df.pos_label]
    unl_idx = decision_df.index[decision_df.labeled == False]
    x_data = decision_df["probability"].values.reshape(-1, 1) 
    y_data = decision_df["pos_labeled"].values.astype(np.int64)
    
    def estimate(x_data, y_data, unl_idx, neg_label_count, pos_num_last):
        regularization_strength = sum(y_data) / neg_label_count 
        lreg = LogisticRegression(
            penalty='l2', 
            fit_intercept=True,
            balanced=True,
            C=regularization_strength)
        lreg.fit(x_data, y_data)
        pos_at = list(lreg.classes_).index(1)
        x_unl = x_data[unl_idx]
        y_pred_proba = lreg.predict_proba(x_unl)[:, pos_at]
        idx_temp_positive = temporary_label_indices(y_pred_proba)
        y_data_new = np.copy(y_data)
        for idx in unl_idx[idx_temp_positive]:
            y_data_new[idx] = 1
        pos_num = sum(y_data_new)
        if pos_num == pos_num_last:
            return pos_num        
        return estimate(x_data, y_data_new, unl_idx, neg_label_count, pos_num)
    r_estimate = estimate(x_data, y_data, unl_idx, neg_label_count, 0)
    return r_estimate
