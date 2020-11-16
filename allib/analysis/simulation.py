


from allib.module.component import Component
from ..instances.base import Instance
from ..analysis.stopping import AbstractStopCriterion
from ..environment.base import AbstractEnvironment
from ..feature_extraction.base import BaseVectorizer
from ..activelearning.base import ActiveLearner
from typing import Any, Dict, List, Tuple, TypeVar

KT = TypeVar("KT")
DT = TypeVar("DT")
VT = TypeVar("VT")
RT = TypeVar("RT")
LT = TypeVar("LT")



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

    # Build the active learners and feature extraction models
    learner: ActiveLearner[KT, DT, VT, RT, LT] = factory.create(
        Component.ACTIVELEARNER, **al_config)
    vectorizer: BaseVectorizer[Instance[KT, DT, np.ndarray, DT]] = factory.create(
        Component.FEATURE_EXTRACTION, **fe_config)
    
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
        if isinstance(learner, Estimator):
            abd = learner.get_abundance(pos_label)
            if abd is not None:
                estimate, _ = abd
        estimates.append(estimate)
        poses.append(pos_count)
        neges.append(neg_count)
    return learner, vectorizer, estimates, poses, neges