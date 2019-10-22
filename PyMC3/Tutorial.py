import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_context('talk')
import pymc3 as pm
from pymc3.distributions.timeseries import GaussianRandomWalk
from scipy import optimize
import pandas as pd

# load data
returns = pd.read_csv('https://raw.githubusercontent.com/pymc-devs/pymc3/master/pymc3/examples/data/SP500.csv', index_col='date')['change']
## data exploration
#fig, ax = plt.subplots(figsize=(14, 8))
#returns.plot(label='S&P500')
#ax.set(xlabel='time', ylabel='returns')
#ax.legend();

with pm.Model() as model:
    step_size = pm.Exponential('step_size', 50.)
    s = GaussianRandomWalk('s', sd=step_size,
                           shape=len(returns))

    nu = pm.Exponential('nu', .1)

    r = pm.StudentT('r', nu=nu,
                    lam=pm.math.exp(-2*s),
                    observed=returns)
with model:
    trace = pm.sample(2000, cores = 1, target_accep = 0.9)

with model:
    pm.traceplot(trace, varnames=['step_size', 'nu'])
    
    fig, ax = plt.subplots()
    plt.plot(trace['s'].T, 'b', alpha=.03)
    ax.set(title=str(s), xlabel='time', ylabel='log volatility')

    fig, ax = plt.subplots(figsize=(14, 8))
    returns.plot(ax=ax)
    ax.plot(np.exp(trace[s].T), 'r', alpha=.03)
    ax.set(xlabel='time', ylabel='returns')
    ax.legend(['S&P500', 'stoch vol'])
