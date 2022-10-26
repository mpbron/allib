from __future__ import annotations

from dataclasses import dataclass
from typing import Any, FrozenSet, Generic, Mapping, OrderedDict, Sequence, Tuple, Dict

import numpy as np
import numpy.typing as npt

from ..typehints.typevars import LT
from ..utils.func import flatten_dicts

from ..activelearning.base import ActiveLearner
from ..activelearning.ensembles import AbstractEnsemble
from .analysis import process_performance

import instancelib as il

@dataclass
class EstimationModelStatistics:
    beta: npt.NDArray[Any]
    mfit: npt.NDArray[Any]
    deviance: float
    preds: npt.NDArray[Any]

@dataclass
class TarDatasetStats:
    pos_count: int
    neg_count: int
    size: int
    prevalence: float

    @classmethod
    def from_learner(cls, 
                     learner: ActiveLearner[Any, Any, Any, Any, Any, LT],
                     pos_label: LT,
                     neg_label: LT) -> TarDatasetStats:
        pos_count = len(learner.env.get_subset_by_labels(learner.env.dataset, pos_label, labelprovider=learner.env.truth))
        neg_count = len(learner.env.get_subset_by_labels(learner.env.dataset, neg_label, labelprovider=learner.env.truth))
        size = len(learner.env.dataset)
        prevalence = pos_count / size
        return TarDatasetStats(pos_count, neg_count, size, prevalence)

@dataclass
class TemporalRecallStats:
    wss: float
    recall: float
    proportional_effort: float
    pos_docs_found: int
    neg_docs_found: int
    effort: int
    loss_er: float
    child_statistics: Sequence[Tuple[str, TemporalRecallStats]]

    @classmethod
    def from_learner(cls,  
                     learner: ActiveLearner[Any, Any, Any, Any, Any, LT],
                     pos_label: LT,
                     neg_label: LT
                     ) -> TemporalRecallStats:
        perf = process_performance(learner, pos_label)
        pos_docs = len(learner.env.get_subset_by_labels(learner.env.labeled, pos_label))
        neg_docs = len(learner.env.get_subset_by_labels(learner.env.labeled, neg_label))
        effort = len(learner.env.labeled)
        prop_effort = effort / len(learner.env.dataset)
        if isinstance(learner, AbstractEnsemble):
            subresults: Sequence[Tuple[str, TemporalRecallStats]] = tuple([
                (str(learner.name), cls.from_learner(learner, pos_label, neg_label)) for learner  in learner.learners])
        else:
            subresults: Sequence[Tuple[str, TemporalRecallStats]] = tuple([])
        return TemporalRecallStats(perf.wss, perf.recall, prop_effort, pos_docs, neg_docs, effort, perf.loss_er, subresults)
    
    def flatten(self, root_name="root") -> Dict[str, TemporalRecallStats]:
        root = {root_name: self}
        children = flatten_dicts(*[stat.flatten(name) for name, stat in self.child_statistics])
        return {**root, **children}

    @staticmethod
    def transpose_dict(recall_stats: Mapping[int, TemporalRecallStats]
                 ) -> Mapping[str, Mapping[int, TemporalRecallStats]]:
        flattened = {key: stat.flatten() for key, stat in recall_stats.items()}
        if flattened:
            learner_names = list(next(iter(flattened.values())).keys())
            ret = {learner: {t: substat[learner] for t,substat in flattened.items()} for learner in learner_names}
            return ret
        return dict()

@dataclass
class DatasetStats(Generic[LT]):
    labels: FrozenSet[LT]

@dataclass
class ALStats:
    unlabeled: int
    labeled: int
    dataset: int

@dataclass
class LabelALStatistics(Generic[LT]):
    label: LT
    seen: int
    generated: int

@dataclass
class BinaryClassificationStatistics(Generic[LT]):
    label: LT
    recall: float
    precision: float
    accuracy: float
    f1: float
