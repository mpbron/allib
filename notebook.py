#%%
import itertools

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from allib.instances import DataPoint
from allib.module.factory import MainFactory, CONFIG
from allib.environment import MemoryEnvironment
from allib import Component
from allib.activelearning import ActiveLearner
from allib.feature_extraction import BaseVectorizer
from allib.activelearning.mostcertain import LabelMaximizer

from allib.module.catalog import ModuleCatalog as Cat

# %%
factory = MainFactory()
# %% DATA IMPORT
dataset = pd.read_csv("../machine-teacher-local/securedata/Software_Engineering_Hall.csv")

#%%
labels = ["Irrelevant", "Relevant"]

# %%
def yield_cols(dataset_df: pd.DataFrame):
    def yield_row_values():
        for i, row in dataset_df.iterrows():
            yield i, str(row["abstract"]), labels[row["included"]]
    indices, texts, labels_true = zip(*yield_row_values())
    return list(indices), list(texts), list(labels_true)
indices_train, texts_train, labels_train = yield_cols(dataset)


#%%
al_config = {
    "paradigm": Cat.AL.Paradigm.POOLBASED,
    "query_type": Cat.AL.QueryType.ESTIMATOR,
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

# %%
al: ActiveLearner = factory.create(Component.ACTIVELEARNER, **al_config)
fe: BaseVectorizer = factory.create(Component.FEATURE_EXTRACTION, **fe_config)
#%%
environment = MemoryEnvironment.from_data(indices_train, texts_train, [], [])
instances = list(environment.dataset_provider.get_all())
matrix = fe.fit_transform(instances)
environment.set_vectors(instances, matrix)
al = al(environment)
# %%
def id_oracle(doc: DataPoint):
    return [labels_train[doc.identifier]]


# %%
def al_loop(learner: ActiveLearner, env: MemoryEnvironment):
    if 1788 in env.unlabeled_provider:
        doc = env.unlabeled_provider[1788]
        env.label_provider.set_labels(doc, "Relevant")
        learner.set_as_labeled(doc)
    if 300 in env.unlabeled_provider:
        doc = env.unlabeled_provider[300]
        env.label_provider.set_labels(doc, "Irrelevant")
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
            env.label_provider.set_labels(instance, *oracle_labels)
            learner.set_as_labeled(instance)
        learner.retrain()
        it = it + 1
al_loop(al, environment)
#%%
al.classifier.predict_proba_instances([environment.dataset_provider[300]])
# %%
test_environment = MemoryEnvironment(indices_test, texts_test, [], labels)
test_instances = test_environment.dataset_provider.get_all()
test_matrix = fe.transform(test_environment.dataset_provider.values())
test_environment.set_vectors(test_instances, test_matrix)
al.predict(test_environment.dataset_provider.values())


# %%
