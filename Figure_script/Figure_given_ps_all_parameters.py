import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# load traces
ts = pickle.load(open("../Data/Syn.pickle", "rb"))
params = ['c', 'p50']
#alpha=0.02, c=8, g1=50, kxmax=7, p50=-4.5
true_values = [8, -4.5]
labels = ['$\\mathit{c}$', '$\\psi_{x50}$ (MPa)']

# figure
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
for i, col in enumerate(ax):
    param = params[i]
    df = pd.DataFrame({param: ts[param]}).iloc[:, 0]
    kde = stats.gaussian_kde(df)
    param_range = np.linspace(np.floor(df.min()), np.ceil(df.max()), 1000)
    col.plot(param_range, kde(param_range), linewidth=2.5, color="black")
    col.axvline(x=true_values[i], c='blue', label='True value')
    col.axes.get_yaxis().set_visible(False)
    col.tick_params(labelsize=20)
    col.set_xlabel(labels[i], fontsize=20)

plt.subplots_adjust(wspace=0.05)
plt.tight_layout
plt.legend(loc='upper left', fontsize=20, framealpha=0)
plt.savefig('../Figures/Figure posterior distributions.png', bbox_inches = 'tight')
