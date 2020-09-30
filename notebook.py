#%%
from typing import List, Tuple, Optional
import itertools

import numpy as np # type: ignore
import pandas as pd # type: ignore

from allib.instances import DataPoint, Instance
from allib.module.factory import MainFactory, CONFIG
from allib.environment import MemoryEnvironment
from allib import Component
from allib.activelearning import ActiveLearner
from allib.feature_extraction import BaseVectorizer
from allib.activelearning.mostcertain import LabelMaximizer

from allib.module.catalog import ModuleCatalog as Cat

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
                "model_configuration": {},
                "task": Cat.ML.Task.BINARY,
                "balancer": {
                    "type": Cat.BL.Type.DOUBLE,
                    "config": {}
                }
            }
        },
        {
            "paradigm": Cat.AL.Paradigm.POOLBASED,
            "query_type": Cat.AL.QueryType.MAX_ENTROPY,
            "machinelearning": {
                "sklearn_model": Cat.ML.SklearnModel.NAIVE_BAYES,
                "model_configuration": {},
                "task": Cat.ML.Task.BINARY,
                "balancer": {
                    "type": Cat.BL.Type.DOUBLE,
                    "config": {}
                }
            }
        },
        {
            "paradigm": Cat.AL.Paradigm.POOLBASED,
            "query_type": Cat.AL.QueryType.RANDOM_SAMPLING,
            "machinelearning": {
                "sklearn_model": Cat.ML.SklearnModel.NAIVE_BAYES,
                "model_configuration": {},
                "task": Cat.ML.Task.BINARY,
                "balancer": {
                    "type": Cat.BL.Type.DOUBLE,
                    "config": {}
                }
            }
        },
    ],
    "machinelearning": {
        "sklearn_model": Cat.ML.SklearnModel.NAIVE_BAYES,
        "model_configuration": {},
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
                "max_features": 300,
                "ngram_range": (1, 1),
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
            yield i, str(row["abstract"]), labels[row["included"]]
    indices, texts, labels_true = zip(*yield_row_values())
    return list(indices), list(texts), list(labels_true)
indices_train, texts_train, labels_train = yield_cols(dataset)

#%%
environment = MemoryEnvironment[int, str, np.ndarray, str].from_data(labels, indices_train, texts_train, [])
instances = environment.dataset.bulk_get_all()
matrix = fe.fit_transform(instances)
environment.set_vectors(instances, matrix)
# %%
al = al(environment)
# %%
def id_oracle(doc: Instance):
    return [labels_train[doc.identifier]]


# %%
def al_loop(learner: ActiveLearner):
    if 1788 in learner.env.unlabeled:
        doc = learner.env.unlabeled[1788]
        al.env.labels.set_labels(doc, "Relevant")
        learner.set_as_labeled(doc)
    if 300 in al.env.unlabeled:
        doc = al.env.unlabeled[300]
        al.env.labels.set_labels(doc, "Irrelevant")
        learner.set_as_labeled(doc)
    learner.retrain()
    count = 0
    it = 1
    while learner.len_labeled < 300:
        sample = itertools.islice(learner, 5)
        for instance in sample:
            oracle_labels = id_oracle(instance)
            if "Relevant" in oracle_labels:
                count += 1
                print(f"Found a relevant document {count} at iteraton {it}")
            al.env.labels.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)
        learner.retrain()
        it = it + 1
al_loop(al)

