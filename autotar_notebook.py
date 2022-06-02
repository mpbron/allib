#%%
from pathlib import Path

from allib.environment.memory import MemoryEnvironment
from sklearn.linear_model import LogisticRegression
from allib.activelearning.autostop import AutoStopLearner
from allib.activelearning.autotar import AutoTarLearner, PseudoInstanceInitializer
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
from allib.estimation.autostop import HorvitzThompsonLoose, HorvitzThompsonVar1, HorvitzThompsonVar2
from allib.module.factory import MainFactory
from allib.stopcriterion.catalog import StopCriterionCatalog
import instancelib as il
from instancelib.ingest.qrel import TrecDataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from allib.stopcriterion.heuristic import AprioriRecallTarget
from allib.stopcriterion.others import BudgetStoppingRule, KneeStoppingRule, ReviewHalfStoppingRule, Rule2399StoppingRule


#%%
# TOPIC_ID = 'CD008081'
# qrel_path = Path("/data/tardata/tr")
# trec = TrecDataset.from_path(qrel_path)
# il_env = trec.get_env('401')
# env = MemoryEnvironment.from_instancelib_simulation(il_env)

#%%

#%%
hall = Path("../instancelib/datasets/Software_Engineering_Hall.csv")
env = read_review_dataset(hall)

#%%
POS = "Relevant"
NEG = "Irrelevant"
init = RandomInitializer(env,1) 
vect = il.TextInstanceVectorizer(
    il.SklearnVectorizer(TfidfVectorizer(stop_words='english', min_df=2, max_features=3000)))
# %%
recall95 = AprioriRecallTarget(POS, 0.95)
recall100 = AprioriRecallTarget(POS,1.0)
knee = KneeStoppingRule(POS)
half = ReviewHalfStoppingRule(POS)
budget = BudgetStoppingRule(POS)
rule2399 = Rule2399StoppingRule(POS)
criteria = {
    "Recall95": recall95, 
    "Recall100": recall100, 
    "Half": half,
    "Knee": knee, 
    "Budget": budget,
    "Rule2399": rule2399,
}
estimators = dict()
classifier = il.SkLearnVectorClassifier.build(
    LogisticRegression(solver="lbfgs", C=1.0, max_iter=10000), 
    env)
at = AutoTarLearner(classifier, POS, NEG, 100, 20)(env)
#%%
init(at)
il.vectorize(vect, at.env)


exp = ExperimentIterator(at, POS, NEG, criteria, estimators)
plotter = TarExperimentPlotter(POS, NEG)
simulator = TarSimulator(exp, plotter, 8800)
# %%
simulator.simulate()
#%%
plotter.show()
#%%
knee.stop_criterion