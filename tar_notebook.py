#%%
import numpy as np
from typing import List
from pathlib import Path
from allib.activelearning.autostop import AutoStopLearner
from allib.activelearning.base import ActiveLearner
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.initialization import RandomInitializer, SeparateInitializer
from allib.analysis.simulation import TarSimulator, initialize
from allib.analysis.tablecollector import TableCollector
from allib.analysis.tarplotter import TarExperimentPlotter
from allib.benchmarking.reviews import read_review_dataset
from allib.configurations.base import (AL_REPOSITORY, ESTIMATION_REPOSITORY,
                                       FE_REPOSITORY, STOP_REPOSITORY)
from allib.configurations.catalog import (ALConfiguration,
                                          EstimationConfiguration,
                                          FEConfiguration)
from allib.estimation.rasch_multiple import FastEMRaschPosNeg, FastOnlyPos, rasch_estimate_parametric
from allib.module.factory import MainFactory
from allib.stopcriterion.catalog import StopCriterionCatalog
import instancelib as il
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

from allib.utils.func import list_unzip


#%%
dis = Path("../datasets/van_Dis_2020.csv")
hall = Path("../instancelib/datasets/Software_Engineering_Hall.csv")
env = read_review_dataset(hall)
POS = "Relevant"
NEG = "Irrelevant"
# %%
# Retrieve the configuration
al_config = AL_REPOSITORY[ALConfiguration("RaschNBLRRF")]
fe_config = FE_REPOSITORY[FEConfiguration("TfIDF5000")]
stop_constructor = STOP_REPOSITORY[StopCriterionCatalog("UpperBound95")]
estimator = FastEMRaschPosNeg(2000)
onlypos = FastOnlyPos(20000)
initializer = SeparateInitializer(env, 1)
factory = MainFactory()

#%%
# Build the experiment objects
al, fe = initialize(factory, al_config, fe_config, initializer, env)
criterion = stop_constructor(estimator, POS)
only_pos_stop = stop_constructor(onlypos, POS)
# %%
criteria =  {"POS and NEG": criterion, "POS": only_pos_stop}# "POS": only_pos_stop}
estimators = {"POS and NEG": estimator, "POS": onlypos} # {"POS": onlypos} #{"POS and NEG": estimator,}# "POS": onlypos}
table_hook = TableCollector(POS)
exp = ExperimentIterator(al, POS, NEG,  criteria, estimators, 
    10, 10, 10, iteration_hooks=[table_hook])
plotter = TarExperimentPlotter(POS, NEG)
simulator = TarSimulator(exp, plotter,500, True)
# %%
simulator.simulate()

#%%
plotter.show()
# %%
table_hook.save_to_folder("results/hall")

# %%

deviances_pn = [info.deviance for info in estimator.model_info]
deviances_p = [info.deviance for info in onlypos.model_info]
plotter.show()
if deviances_pn:
    plt.plot(range(0,plotter.it,10), deviances_pn, label="Deviance POS and NEG", linestyle="--")
if deviances_p:
    plt.plot(range(0,plotter.it,10), deviances_p, label="Deviance POS", linestyle="--")
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
# %%

df = estimator.dfs[-2]
# %%
rasch_estimate_parametric(df, 8911, multinomial_size=20)
# %%
from allib.estimation.rasch_multiple import rasch_parallel
rasch_parallel.inspect_types()
# %%
df.to_csv("problem.csv",index=False)
# %%
