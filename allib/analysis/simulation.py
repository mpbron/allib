


import itertools
import random
import numpy as np
from typing import Any, Dict, List, Sequence, Tuple, TypeVar

from ..activelearning.base import ActiveLearner
from ..analysis.analysis import process_performance
from ..analysis.stopping import AbstractStopCriterion
from ..environment.base import AbstractEnvironment
from ..environment.memory import MemoryEnvironment
from ..feature_extraction.base import BaseVectorizer
from ..instances.base import Instance
from ..module.component import Component
from ..utils.chunks import divide_sequence

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")

def add_doc_from_truth(learner: ActiveLearner[KT, DT, VT, RT, LT], id: KT):
    doc = learner.env.unlabeled[id]
    labels = learner.env.truth.get_labels(doc)
    learner.env.labels.set_labels(doc, *labels)
    learner.set_as_labeled(doc)

def vectorize(vectorizer: BaseVectorizer[Instance[KT, DT, np.ndarray, DT]], 
              environment: MemoryEnvironment[KT, DT, np.ndarray, LT]) -> None:
    instances = environment.dataset.bulk_get_all()
    vectorizer.fit(instances)
    def set_vectors(instances: Sequence[Instance[KT, DT, np.ndarray, DT]], batch_size = 400) -> None:
        instance_chunks = divide_sequence(instances, batch_size)
        for instance_chunk in instance_chunks:
            matrix = vectorizer.transform(instance_chunk)
            for i, instance in enumerate(instance_chunk):
                instance.vector = matrix[i,:]
    set_vectors(instances)

SimulationResult = Tuple[ActiveLearner[KT, DT, VT, RT, LT],
                         BaseVectorizer[Instance[KT, DT, VT, RT]],
                         List[float],
                         List[int],
                         List[int]]

def simulate(learner: ActiveLearner[KT, DT, VT, RT, LT], 
             vectorizer: BaseVectorizer[Instance[KT, DT, VT, RT]], 
             start_env: AbstractEnvironment[KT, DT, VT, RT, LT], 
             stop_crit: AbstractStopCriterion,
             pos_label: LT, neg_label: LT,
             batch_size: int, 
             target_recall: float = 0.95) -> SimulationResult:

    ## Copy the data to memory
    env = MemoryEnvironment.from_environment_only_data(start_env)

    # Vectorize the dataset
    vectorize(vectorizer, env)

    # Attach the environment to the active learner
    learner = learner(env)
    
    # Sample 1 pos and 1 neg document
    pos_docs = random.sample(learner.env.truth.get_instances_by_label(pos_label), 1)
    neg_docs = random.sample(learner.env.truth.get_instances_by_label(neg_label), 1)
    docs = pos_docs + neg_docs

    # Add the preliminary labeled documents to the dataset
    for doc in docs:
        add_doc_from_truth(learner, doc)
    
    estimates: List[float] = []
    poses: List[int] = []
    neges: List[int] = []
    oracle_count = 0
    while not stop_crit.stop_criterion:
        # Train the model
        learner.retrain()
        # Sample batch_size documents from the learner
        sample = itertools.islice(learner, batch_size)
        for instance in sample:
            # Retrieve the labels from the oracle
            oracle_labels = learner.env.truth.get_labels(instance)
            oracle_count += 1

            # Set the labels for the document
            learner.env.labels.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)

            # Stop within this loop as soon as target recall has been hit
            if process_performance(learner, pos_label).recall >= target_recall:
                break
        stop_crit.update(learner)
        # Estimate the number of remaining positive documents
        pos_count = learner.env.labels.document_count(pos_label)
        neg_count = learner.env.labels.document_count(neg_label)
        stop_crit.update(learner)
        estimate = 0.0
        estimates.append(estimate)
        poses.append(pos_count)
        neges.append(neg_count)
    return learner, vectorizer, estimates, poses, neges
