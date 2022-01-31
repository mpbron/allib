#%%
from pathlib import Path
from allib.activelearning.autostop import AutoStopLearner
from allib.activelearning.autotar import AutoTarLearner
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.initialization import RandomInitializer, SeparateInitializer
from allib.analysis.simulation import TarSimulator, initialize
from allib.analysis.tarplotter import TarExperimentPlotter
from allib.benchmarking.reviews import read_review_dataset
from allib.configurations.base import (AL_REPOSITORY, ESTIMATION_REPOSITORY,
                                       FE_REPOSITORY, STOP_REPOSITORY)
from allib.configurations.catalog import (ALConfiguration,
                                          EstimationConfiguration,
                                          FEConfiguration)
from allib.module.factory import MainFactory
from allib.stopcriterion.catalog import StopCriterionCatalog
import instancelib as il
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer


#%%
path = Path("../instancelib/datasets/Software_Engineering_Hall.csv")
env = read_review_dataset(path)
POS = "Relevant"
NEG = "Irrelevant"
vect = il.TextInstanceVectorizer(
    il.SklearnVectorizer(TfidfVectorizer(max_features=5000)))
il.vectorize(vect, env)
# %%
classifier = il.SkLearnVectorClassifier.build(MultinomialNB(), env)
al = AutoStopLearner(classifier, POS, NEG, 100, 20)(env)
at = AutoTarLearner(classifier, POS, NEG, 100, 20)(env)
random_init = RandomInitializer(env)
random_init(at)
exp = ExperimentIterator(at, POS, NEG, {}, {})
plotter = TarExperimentPlotter(POS, NEG)
simulator = TarSimulator(exp, plotter, 8800)
# %%
simulator.simulate()
# %%
