#%%
from pathlib import Path
from allib.activelearning.tsvm import TSVMLearner
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.initialization import RandomInitializer
from allib.analysis.simulation import TarSimulator
from allib.analysis.tarplotter import TarExperimentPlotter
from allib.benchmarking.reviews import read_review_dataset
import instancelib as il
from sklearn.feature_extraction.text import TfidfVectorizer

from allib.stopcriterion.heuristic import AprioriRecallTarget

#%%
path = Path("../instancelib/datasets/Software_Engineering_Hall.csv")
env = read_review_dataset(path)
POS = "Relevant"
NEG = "Irrelevant"
vect = il.TextInstanceVectorizer(
    il.SklearnVectorizer(TfidfVectorizer(max_features=5000))
)
il.vectorize(vect, env)
# %%
recall95 = AprioriRecallTarget(POS, 0.95)
criteria = {"Recall95": recall95}
# estimators = {"RaschRidge": estimator}
# %%
at = TSVMLearner.builder(POS, NEG, 200, 200)(env)
random_init = RandomInitializer(100)
random_init(at)

#%%
at.classifier.fit_provider(at.env.dataset, at.env.labels)
# %%
exp = ExperimentIterator(at, POS, NEG, criteria, {})
plotter = TarExperimentPlotter(POS, NEG)
simulator = TarSimulator(exp, plotter, 300)
#%%
simulator.simulate()
# %%
not_found_yet = at.env.get_subset_by_labels(
    at.env.unlabeled, "Relevant", labelprovider=at.env.truth
)
# %%
