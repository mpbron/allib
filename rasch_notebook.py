#%%
import pandas as pd
from allib.estimation.rasch_multiple import *

#%%
df = pd.read_csv("problem.csv")
# %%
estimate, stats = rasch_estimate_parametric(df, 8911, multinomial_size=2000)
# %%
