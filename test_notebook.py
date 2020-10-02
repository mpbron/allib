#%%
from allib.environment.base import AbstractEnvironment
import logging
from typing import List, Tuple, Any, Dict
import itertools
import random

import numpy as np # type: ignore
import pandas as pd # type: ignore

from allib.instances import  Instance
from allib.module.factory import MainFactory
from allib.environment import MemoryEnvironment
from allib import Component
from allib.activelearning import ActiveLearner
from allib.feature_extraction import BaseVectorizer


from allib.module.catalog import ModuleCatalog as Cat

# %%
# create logger
logger = logging.getLogger("allib")
logger.setLevel(logging.DEBUG)

# create console handler and set level to debug
ch = logging.StreamHandler()
ch.setLevel(logging.DEBUG)

# create formatter
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

# add formatter to ch
ch.setFormatter(formatter)

# add ch to logger

# %%
factory = MainFactory()
#%%
al_config = {
    "paradigm": Cat.AL.Paradigm.ESTIMATOR,
    "learners": [
        {
            "paradigm": Cat.AL.Paradigm.POOLBASED,
            "query_type": Cat.AL.QueryType.LABELMAXIMIZER,
            "label": "Relevant",
            "machinelearning": {
                "sklearn_model": Cat.ML.SklearnModel.NAIVE_BAYES,
                "model_configuration": {"alpha": 3.822},
                "task": Cat.ML.Task.BINARY,
                "balancer": {
                    "type": Cat.BL.Type.DOUBLE,
                    "config": {}
                }
            }
        },
        # {
        #     "paradigm": Cat.AL.Paradigm.POOLBASED,
        #     "query_type": Cat.AL.QueryType.MAX_ENTROPY,
        #     "machinelearning": {
        #         "sklearn_model": Cat.ML.SklearnModel.RANDOM_FOREST,
        #         "model_configuration": {},
        #         "task": Cat.ML.Task.BINARY,
        #         "balancer": {
        #             "type": Cat.BL.Type.DOUBLE,
        #             "config": {}
        #         }
        #     }
        # }
    ],
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.NAIVE_BAYES,
        "model_configuration": {"alpha": 3.822},
        "task": Cat.ML.Task.BINARY,
        "balancer": {
            "type": Cat.BL.Type.DOUBLE,
            "config": {}
        }
    }
}
env_config = { "environment_type": Cat.ENV.Type.MEMORY }

fe_config ={
    "datatype": Cat.FE.DataType.TEXTINSTANCE,
    "vec_type": Cat.FE.VectorizerType.STACK,
    "vectorizers": [
        {
            "vec_type": Cat.FE.VectorizerType.SKLEARN,
            "sklearn_vec_type": Cat.FE.SklearnVecType.TFIDF_VECTORIZER,
            "sklearn_config": {
                "max_features": 5000
            }
        }
    ]
}

# %% Create components
al: ActiveLearner = factory.create(Component.ACTIVELEARNER, **al_config)
fe: BaseVectorizer = factory.create(Component.FEATURE_EXTRACTION, **fe_config)
# %% DATA IMPORT
dataset = pd.read_csv("./datasets/Software_Engineering_Hall.csv")

#%%
labels = ["Irrelevant", "Relevant"]

# %%
def yield_cols(dataset_df: pd.DataFrame) -> Tuple[List[int], List[str], List[str]]:
    def yield_row_values():
        for i, row in dataset_df.iterrows():
            yield int(i), str(row["abstract"]), labels[row["included"]] # type: ignore
    indices, texts, labels_true = zip(*yield_row_values())
    return list(indices), list(texts), list(labels_true)
indices_train, texts_train, labels_train = yield_cols(dataset)

#%%
def get_labels_idx(idx: List[int], lbl: List[str]) -> Dict[str,List[int]]:
    ret = [
        (
            label, [
                idx  for (idx, lbl) in zip(idx, lbl) if lbl == label
            ]
        )
        for label in set(lbl)
    ]
    return dict(ret)
label_idx = get_labels_idx(indices_train, labels_train)
#%%
environment = MemoryEnvironment[int, str, np.ndarray, str].from_data(labels, indices_train, texts_train, [])
instances = environment.dataset.bulk_get_all()
matrix = fe.fit_transform(instances)
environment.set_vectors(instances, matrix)
# %%
def id_oracle(doc: Instance):
    return [labels_train[doc.identifier]]

def add_doc(learner: ActiveLearner, id):
    if id in learner.env.unlabeled:
        doc = learner.env.unlabeled[id]
        labels = id_oracle(doc)
        learner.env.labels.set_labels(doc, *labels)
        learner.set_as_labeled(doc)
    else:
        logger.debug("Already labeled")
# %%
def al_loop(learner: ActiveLearner, start_env: AbstractEnvironment, label_dict, pos_label, neg_label, batch_size):
    ## Initialize new environment
    learner = learner(MemoryEnvironment.from_environment(start_env, shared_labels=False))
    
    # Sample 2 pos en 5 neg documents
    pos_docs = random.sample(label_dict[pos_label], 1)
    neg_docs = random.sample(label_dict[neg_label], 1)
    docs = pos_docs + neg_docs
    for doc in docs:
        add_doc(learner, doc)
    
    # Train the model
    learner.retrain()
    
    # Start the active learning loop
    count = 0
    it = 1
    while len(learner.env.labels.get_instances_by_label("Relevant")) < 40:
        sample = itertools.islice(learner, batch_size)
        for instance in sample:
            oracle_labels = id_oracle(instance)
            if "Relevant" in oracle_labels:
                count += 1
                print(f"Found document {count} after reading {it} documents")
            learner.env.labels.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)
            it = it + 1
        learner.retrain()
# %%
al_loop(al, environment, label_idx, "Relevant", "Irrelevant", 5)


# %%
