#%%
from pathlib import Path

from allib.environment.memory import MemoryEnvironment
from sklearn.linear_model import LogisticRegression
from allib.activelearning.autostop import AutoStopLearner
from allib.activelearning.autotar import AutoTarLearner, PseudoInstanceInitializer
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.initialization import RandomInitializer, SeparateInitializer
from allib.analysis.simulation import TarSimulator, initialize_tar_simulation
from allib.analysis.tarplotter import TarExperimentPlotter
from allib.benchmarking.reviews import read_review_dataset
from allib.configurations.catalog import (
    ALConfiguration,
    EstimationConfiguration,
    FEConfiguration,
)
from allib.estimation.autostop import (
    HorvitzThompsonLoose,
    HorvitzThompsonVar1,
    HorvitzThompsonVar2,
)
from allib.module.factory import MainFactory
from allib.stopcriterion.catalog import StopCriterionCatalog
import instancelib as il
from instancelib.ingest.qrel import TrecDataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from allib.stopcriterion.estimation import Conservative, Optimistic

from allib.stopcriterion.heuristic import AprioriRecallTarget
from allib.stopcriterion.others import KneeStoppingRule


#%%
# TOPIC_ID = 'CD008081'
# qrel_path = Path("/data/tardata/tr")
# trec = TrecDataset.from_path(qrel_path)
# il_env = trec.get_env('401')
# env = MemoryEnvironment.from_instancelib_simulation(il_env)

#%%

#%%
wolters = Path("../datasets/Wolters_2018.csv")
ace = Path("../datasets/ACEInhibitors.csv")
dis = Path("../datasets/van_Dis_2020.csv")
schoot = Path("../datasets/PTSD_VandeSchoot_18.csv")
hall = Path("../datasets/Software_Engineering_Hall.csv")
nudging = Path("../datasets/Nagtegaal_2019.csv")
bos = Path("../datasets/Bos_2018.csv")
wilson = Path("../datasets/Appenzeller-Herzog_2020.csv")
bb = Path("../datasets/Bannach-Brown_2019.csv")
wolters = Path("../datasets/Wolters_2018.csv")
virus = Path("../datasets/Kwok_2020.csv")
env = read_review_dataset(ace)

#%%
POS = "Relevant"
NEG = "Irrelevant"
init = RandomInitializer(1)
vect = il.TextInstanceVectorizer(
    il.SklearnVectorizer(
        TfidfVectorizer(stop_words="english", min_df=2, max_features=3000)
    )
)
# %%
recall95 = AprioriRecallTarget(POS, 0.95)
recall100 = AprioriRecallTarget(POS, 1.0)
knee = KneeStoppingRule(POS)
hvt = HorvitzThompsonVar2()
estimators = {"HorvitzThompson2": hvt}
criteria = {
    "Recall95": recall95,
    "Recall100": recall100,
    "KneeMethod": knee,
    "AutoStop-CON": Conservative(hvt, POS, 0.95),
    "AutoStop-OPT": Optimistic(hvt, POS, 1),
}
classifier = lambda env: il.SkLearnVectorClassifier.build(
    LogisticRegression(solver="lbfgs", C=1.0, max_iter=10000), env
)
al = AutoStopLearner.builder(classifier, 100, 1)(env, pos_label=POS, neg_label=NEG)
#%%
init(al)
il.vectorize(vect, al.env)

# at = AutoTarLearner(classifier, POS, NEG, 100, 20)(env)
exp = ExperimentIterator(al, POS, NEG, criteria, estimators)
plotter = TarExperimentPlotter(POS, NEG)
simulator = TarSimulator(exp, plotter, 8800)
# %%
simulator.simulate()
#%%
plotter.show()
#%%
knee.stop_criterion
