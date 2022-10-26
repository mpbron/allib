#%%
import pickle
from allib.analysis.tarplotter import TarExperimentPlotter
#%%
with open("vandis.pkl", "rb") as fh:
    plotter: TarExperimentPlotter = pickle.load(fh)
# %%
plotter.estimates = dict()
plotter.stop_results = dict()
# %%
plotter.show()