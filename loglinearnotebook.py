#%%
import numpy as np
from typing import List
from pathlib import Path
from instancelib.ingest.qrel import TrecDataset
from allib.activelearning.autostop import AutoStopLearner
from allib.activelearning.base import ActiveLearner
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.initialization import RandomInitializer, SeparateInitializer
from allib.analysis.simulation import TarSimulator, initialize
from allib.analysis.tablecollector import TableCollector
from allib.analysis.tarplotter import ModelStatsTar, TarExperimentPlotter
from allib.benchmarking.reviews import read_review_dataset
from allib.configurations.base import (AL_REPOSITORY, ESTIMATION_REPOSITORY,
                                       FE_REPOSITORY, STOP_REPOSITORY)
from allib.configurations.catalog import (ALConfiguration,
                                          EstimationConfiguration,
                                          FEConfiguration)
from allib.environment.memory import MemoryEnvironment
from allib.estimation.rasch_multiple import FastEMRaschPosNeg, FastOnlyPos, FastPosAssisted, rasch_estimate_parametric
from allib.estimation.loglinear import LogLinear
from allib.module.factory import MainFactory
from allib.stopcriterion.catalog import StopCriterionCatalog
import instancelib as il
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt
from allib.estimation.mhmodel import AbundanceEstimator
from allib.stopcriterion.heuristic import AprioriRecallTarget
from allib.stopcriterion.others import BudgetStoppingRule, KneeStoppingRule, ReviewHalfStoppingRule, Rule2399StoppingRule, StopAfterKNegative
from allib.utils.func import list_unzip


#%%
TOPIC_ID = 'CD008081'
qrel_path = Path("/data/tardata/tr")
#trec = TrecDataset.from_path(qrel_path)
#il_env = trec.get_env('401')
#env = MemoryEnvironment.from_instancelib_simulation(il_env)
wolters = Path("../datasets/Wolters_2018.csv")
dis = Path("../datasets/van_Dis_2020.csv")
schoot = Path("../datasets/PTSD_VandeSchoot_18.csv")
hall = Path("../datasets/Software_Engineering_Hall.csv")
nagtegaal = Path("../datasets/Nagtegaal_2019.csv")
bos = Path("../datasets/Bos_2018.csv")
ah = Path("../datasets/Appenzeller-Herzog_2020.csv")
bb = Path("../datasets/Bannach-Brown_2019.csv")
wolters = Path("../datasets/Wolters_2018.csv")
kwok = Path("../datasets/Kwok_2020.csv")
env = read_review_dataset(dis)
POS = "Relevant"
NEG = "Irrelevant"
# %%
# Retrieve the configuration
al_config = AL_REPOSITORY[ALConfiguration.RaschNBLRRFLGBMRAND]
fe_config = FE_REPOSITORY[FEConfiguration("TfIDF5000")]
stop_constructor = STOP_REPOSITORY[StopCriterionCatalog("UpperBound95")]
chao = AbundanceEstimator()
logl = LogLinear(2000)
rasch = FastOnlyPos(2000)
initializer = SeparateInitializer(env, 1)
factory = MainFactory()

#%%
# Build the experiment objects
al, fe = initialize(factory, al_config, fe_config, initializer, env)
chao_stop = stop_constructor(chao, POS)
recall95 = AprioriRecallTarget(POS, 0.95)
recall100 = AprioriRecallTarget(POS, 1.0)
knee = KneeStoppingRule(POS)
half = ReviewHalfStoppingRule(POS)
budget = BudgetStoppingRule(POS)
rule2399 = Rule2399StoppingRule(POS)
stop200 = StopAfterKNegative(POS, 200)
stop400 = StopAfterKNegative(POS, 400)
criteria = {
    "Recall95": recall95, 
    "Recall100": recall100, 
    "Half": half,
    "Knee": knee, 
    "Budget": budget,
    "Rule2399": rule2399,
    "Stop200": stop200,
    "Stop400": stop400,
    "Mh Chao LB": chao_stop
}
# %%
estimators = {"Mh Chao LB": chao} # {"POS": onlypos} #{"POS and NEG": estimator,}# "POS": onlypos}
table_hook = TableCollector(POS)
exp = ExperimentIterator(al, POS, NEG,  criteria, estimators, 
    10, 10, 10, iteration_hooks=[table_hook])
plotter = ModelStatsTar(POS, NEG)
simulator = TarSimulator(exp, plotter, 1000)
# %%
#simulator.max_it += 1000
simulator.simulate()

#%%
plotter.show()
# %%
def save_to_folder(table_hook, path: "PathLike[str]") -> None:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    #compact = self.compact
    #compact.to_csv((path / "aggregated.csv"), index=False)
    for i, df in enumerate(table_hook.dfs):
        df.to_csv((path / f"design_matrix_{i}.csv"), index=False)


# %%
cd 