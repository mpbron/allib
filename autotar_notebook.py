#%%
from pathlib import Path
from allib.activelearning.factory import classifier_builder

from allib.environment.memory import MemoryEnvironment
from sklearn.linear_model import LogisticRegression
from allib.activelearning.autostop import AutoStopLearner
from allib.activelearning.autotar import AutoTarLearner, PseudoInstanceInitializer
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.initialization import RandomInitializer, SeparateInitializer
from allib.analysis.simulation import TarSimulator, initialize_tar_simulation
from allib.analysis.tarplotter import TarExperimentPlotter
from allib.benchmarking.reviews import read_review_dataset
import instancelib as il
from instancelib.ingest.qrel import TrecDataset
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

from allib.stopcriterion.heuristic import AprioriRecallTarget
from allib.stopcriterion.others import (
    BudgetStoppingRule,
    KneeStoppingRule,
    ReviewHalfStoppingRule,
    Rule2399StoppingRule,
    StopAfterKNegative,
)


#%%
TOPIC_ID = "CD008081"
qrel_path = Path("/data/tardata/tr")
trec = TrecDataset.from_path(qrel_path)
il_env = trec.get_env("401")
env = MemoryEnvironment.from_instancelib_simulation(il_env)

#%%

#%%
# wolters = Path("../datasets/Wolters_2018.csv")
# ace = Path("../datasets/ACEInhibitors.csv")
# dis = Path("../datasets/van_Dis_2020.csv")
# schoot = Path("../datasets/PTSD_VandeSchoot_18.csv")
# hall = Path("../datasets/Software_Engineering_Hall.csv")
# nudging = Path("../datasets/Nagtegaal_2019.csv")
# bos = Path("../datasets/Bos_2018.csv")
# wilson = Path("../datasets/Appenzeller-Herzog_2020.csv")
# bb = Path("../datasets/Bannach-Brown_2019.csv")
# wolters = Path("../datasets/Wolters_2018.csv")
# virus = Path("../datasets/Kwok_2020.csv")
# env = read_review_dataset(bos)

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
half = ReviewHalfStoppingRule(POS)
budget = BudgetStoppingRule(POS)
rule2399 = Rule2399StoppingRule(POS)
stop200 = StopAfterKNegative(POS, 200)
stop400 = StopAfterKNegative(POS, 400)
criteria = {
    "Perfect95": recall95,
    "Perfect100": recall100,
    "Half": half,
    "Knee": knee,
    "Budget": budget,
    "Rule2399": rule2399,
    "Stop200": stop200,
    "Stop400": stop400,
}
estimators = dict()
logreg = LogisticRegression(solver="lbfgs", C=1.0, max_iter=10000)
classifier = classifier_builder(logreg, il.SkLearnVectorClassifier.build)
at = AutoTarLearner.builder(classifier, 100, 20)(env, POS, NEG)
# %%
init(at)
il.vectorize(vect, at.env)

#%%
not at.env.unlabeled
# %%

exp = ExperimentIterator(at, POS, NEG, criteria, estimators)
plotter = TarExperimentPlotter(POS, NEG)
simulator = TarSimulator(exp, plotter, 1000)
# %%
simulator.simulate()
#%%
plotter.show()
#%%
from explabox import Explabox
from instancelib.machinelearning.autovectorizer import AutoVectorizerClassifier

box = Explabox(
    data=at.env, model=AutoVectorizerClassifier.from_skvector(at.classifier, vect)
)
# %%
doc = next(at)
box.explain.explain_prediction(doc, methods=["LIME"], n_features=20)
# %%

# %%
