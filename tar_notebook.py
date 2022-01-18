#%%
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.initialization import SeparateInitializer
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

#%%
env = read_review_dataset("../instancelib/datasets/Software_Engineering_Hall.csv")
POS = "Relevant"
NEG = "Irrelevant"
# %%
# Retrieve the configuration
al_config = AL_REPOSITORY[ALConfiguration("RaschEstimator")]
fe_config = FE_REPOSITORY[FEConfiguration("TfIDF5000")]
stop_constructor = STOP_REPOSITORY[StopCriterionCatalog("UpperBound95")]
estimator = ESTIMATION_REPOSITORY[EstimationConfiguration("RaschRidge")]
initializer = SeparateInitializer(env, 1)
factory = MainFactory()

#%%
# Build the experiment objects
al, fe = initialize(factory, al_config, fe_config, initializer, env)
criterion = stop_constructor(estimator, POS)
# %%
criteria = {"Upperbound95": criterion}
estimators = {"RaschRidge": estimator}
exp = ExperimentIterator(al, POS, NEG,  criteria, estimators, 10, 10, 10)
plotter = TarExperimentPlotter(POS, NEG)
simulator = TarSimulator(exp, plotter, 500)
# %%
simulator.simulate()
# %%
plotter.show()
# %%
