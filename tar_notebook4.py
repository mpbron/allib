#%%
import numpy as np
from typing import List
from pathlib import Path
from allib.analysis.experiments import ExperimentIterator
from allib.analysis.initialization import SeparateInitializer
from allib.analysis.simulation import TarSimulator, initialize_tar_simulation
from allib.analysis.tablecollector import TableCollector
from allib.analysis.tarplotter import ModelStatsTar
from allib.benchmarking.reviews import read_review_dataset
from allib.configurations.base import (
    AL_REPOSITORY,
    FE_REPOSITORY,
)
from allib.configurations.catalog import (
    ALConfiguration,
    FEConfiguration,
)
from allib.estimation.rasch_multiple import (
    FastEMRaschPosNeg,
)
from allib.module.factory import MainFactory
from allib.stopcriterion.estimation import Conservative

from allib.utils.func import list_unzip


#%%
# TOPIC_ID = 'CD008081'
# qrel_path = Path("/data/tardata/tr")
# trec = TrecDataset.from_path(qrel_path)
# il_env = trec.get_env('401')
# env = MemoryEnvironment.from_instancelib_simulation(il_env)
# dis = Path("../datasets/van_Dis_2020.csv")
hall = Path("../instancelib/datasets/Software_Engineering_Hall.csv")
env = read_review_dataset(hall)
POS = "Relevant"
NEG = "Irrelevant"
# %%
# Retrieve the configuration
al_config = AL_REPOSITORY[ALConfiguration.RaschNBLRRFSVM]
fe_config = FE_REPOSITORY[FEConfiguration("TfIDF5000")]
stop_constructor = lambda est, lbl: Conservative(est, lbl, 0.95)
onlypos = FastEMRaschPosNeg(2000)
initializer = SeparateInitializer(1)
factory = MainFactory()

#%%
# Build the experiment objects
al, fe = initialize_tar_simulation(
    factory, al_config, fe_config, initializer, env, POS, NEG
)
only_pos_stop = stop_constructor(onlypos, POS)
# %%
criteria = {"POS": only_pos_stop}  # "POS": only_pos_stop}
estimators = {
    "POS": onlypos
}  # {"POS": onlypos} #{"POS and NEG": estimator,}# "POS": onlypos}
table_hook = TableCollector(POS)
exp = ExperimentIterator(
    al, POS, NEG, criteria, estimators, 10, 10, 10, iteration_hooks=[table_hook]
)
plotter = ModelStatsTar(POS, NEG)
simulator = TarSimulator(exp, plotter, 400, True)
# %%
simulator.simulate()

#%%
plotter.show()
# %%
def save_to_folder(table_hook, path: Path) -> None:
    path = Path(path)
    if not path.exists():
        path.mkdir(parents=True)
    # compact = self.compact
    # compact.to_csv((path / "aggregated.csv"), index=False)
    for i, df in enumerate(table_hook.dfs):
        df.to_csv((path / f"design_matrix_{i}.csv"), index=False)


# %%
