#%%
import pymc3 as pm
from theano import tensor as tt
import pandas as pd
import numpy as np
import numpy.typing as npt
import arviz as az
from scipy.special import binom
#%%
data = pd.read_csv("results/history2.csv")
data = data.set_index("Unnamed: 0")


#%%
alpha = 1
eta = 1
mu = 0
taub = 10
n_dataset = 8911

#
N = 104
K = 3

with pm.Model() as fienberg_rasch:
    # Hyper parameters catchability
    α = 1
    η = 1
    
    # Hyper parameters list vector
    τ_β = 10
    μ_β = 0
    
    
    σ_θ = pm.InverseGamma("σ_θ", α, η)
    
    # Rasch Priors
    Θ = pm.Normal("θ", 
                  0, σ_θ, 
                  shape=(1, N)) ## F_Θ(Θ_i), i = 1, ... , N
    β = pm.Normal("β", 
                  μ_β, τ_β, 
                  shape=(K, 1)) ## G_β(β_j), j = 1, ... , J

    
    X = pm.Bernoulli("y", logit_p=η, observed=data.values)

#%%
with pm.Model() as model:

    # Data
    data = pm.Data("observed", df.foul_called)

    # Hyperpriors
    μ_θ = pm.Normal("μ_θ", 0.0, 100.0)
    σ_θ = pm.HalfCauchy("σ_θ", 2.5)
    σ_b = pm.HalfCauchy("σ_b", 2.5)

    # Priors
    Δ_θ = pm.Normal("Δ_θ", 0.0, 1.0, dims="disadvantaged")
    Δ_b = pm.Normal("Δ_b", 0.0, 1.0, dims="committing")

    # Deterministic
    θ = pm.Deterministic("θ", Δ_θ * σ_θ + μ_θ, dims="disadvantaged")
    b = pm.Deterministic("b", Δ_b * σ_b, dims="committing")
    η = pm.Deterministic("η", tt.log(θ + b)

    # Likelihood
    y = pm.Bernoulli("y", logit_p=η, observed=data.values)

#%%
with pm.Model() as model:
    sigma2 = pm.InverseGamma("Sigma2", alpha,eta)
    
    # General statistics    
    n_members = data.shape[1]
    n_seen = len(data)
    
    
    N = 104
    n_missing =  N - n_seen 

    # Rasch priors
    theta = pm.Normal("Document", mu = mu, sigma = taub, 
        shape = (1, N))
    beta = pm.Normal("Member", mu = 0, sigma=sigma2, 
        shape=(n_members, 1))

    def logp(d_seen: npt.NDArray[Any]):
        # Create vectors for unseen values
        d_missing = tt.zeros((n_missing, n_members))
        mat = tt.concatenate((d_seen, d_missing))
        v1 = tt.transpose(mat) * tt.log(pm.math.sigmoid(theta - (beta - beta.mean(0)))) # type: ignore
        v2 = tt.transpose((1-mat)) * tt.log(1 - pm.math.sigmoid(theta - (beta - beta.mean(0)))) # type: ignore
        return v1 + v2
    
    ll = pm.DensityDist('ll', logp, observed = {'d_seen': data.values})
    trace = pm.sample(3000, cores=-1, step = pm.NUTS(), return_inferencedata=True)

#%%
with model:
    pm.plot_energy(trace)
    pm.plot_trace(trace)
# %%
with model:
    az.plot_forest(
        trace,
        var_names=["Member"],
        combined=True,
    )
# %%
