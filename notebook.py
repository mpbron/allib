#%%
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from environment import MemoryEnvironment
from factory.base import FACTORY, CONFIG
from factory.catalog import *
from activelearning import ActiveLearner, RandomSampling
from activelearning.mostcertain import LabelMaximizer
from instances.memory import DataPoint
from feature_extraction.base import BaseVectorizer

# %%
CONFIG = {
    "al_paradigm": ALParadigm.POOLBASED,
    "query_type": QueryType.INTERLEAVE,
    "environment_type": EnvironmentType.MEMORY,
    "datatype": DataType.TEXTINSTANCE,
    "vec_type": VectorizerType.STACK,
    "vectorizers": [
        {
            "vec_type": VectorizerType.SKLEARN,
            "sklearn_vec_type": SklearnVecType.TFIDF_VECTORIZER,
            "sklearn_config": {
                "max_features": 400,
                "ngram_range": (1,1)
            },
            "storage_location": "../securedata/",
            "filename": "vectorizer.pkl"
        }
    ],
    "ml_task": MachineLearningTask.N,
    "ml_model": {
        "ml_task:"
    }
    "sklearn_model": SklearnModel.RANDOM_FOREST,
    "model_configuration": {},
    
    "storage_location": "../securedata/",
    "filename": "model.pkl"
}


# %% DATA IMPORT
dataset = pd.read_csv("../securedata/Software_Engineering_Hall.csv")

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

# %%
environment = MemoryEnvironment(indices_train, texts_train, [], labels)
al: RandomSampling = FACTORY.create(Component.ACTIVELEARNER, **CONFIG)
fe: BaseVectorizer = FACTORY.create(Component.FEATURE_EXTRACTION, **CONFIG)
#%%
instances = list(environment.dataset_provider.get_all())
matrix = fe.fit_transform(instances)
environment.set_vectors(instances, matrix)

#%% 
# Attach environment
al = LabelMaximizer(al.classifier, "Relevant")
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
        sample = [next(learner)]
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
