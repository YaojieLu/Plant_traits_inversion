import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

### load traces
ts1 = pickle.load(open("../Data/baseline.pickle", "rb"))['p50']# traces baseline
ts2 = pickle.load(open("../Data/given_L.pickle", "rb"))['p50']# traces given L
ts3 = pickle.load(open("../Data/given_ps.pickle", "rb"))['p50']# traces given ps
df = pd.DataFrame({'ts1': ts1, 'ts2':ts2, 'ts3':ts3})

# figure
labels = ['Unknown $\psi_{s}$ and leaf area', 'Given leaf area', 'Given $\psi_{s}$']
fig = plt.figure(figsize = (20, 6))
for i in range(len(df.columns)):
    ax = fig.add_subplot(1, len(df.columns), i+1)
    #n, bins, patches = plt.hist(df.iloc[:, i], bins = 20, normed = True,
    #                            color = "black")
    kde = stats.gaussian_kde(df.iloc[:, i])
    xx = np.linspace(-10, 0, 1000)
    ax.plot(xx, kde(xx), linewidth = 2.5, color = "black")
    plt.axvline(x = -4.5, c = 'blue', label='True value')
    ax.axes.get_yaxis().set_visible(False)
    ax.tick_params(labelsize = 20)
    plt.xlabel('$\psi_{x50}$ (MPa)', fontsize = 20)
    plt.title(labels[i], fontsize = 20)       

plt.subplots_adjust(top = 0.83, wspace = 0.08)
plt.tight_layout
fig.suptitle('Posterior distribution for $\psi_{x50}$', fontsize = 30)
ax.legend(loc = 'upper left', fontsize = 20, framealpha = 0)
plt.savefig('../Figures/Figure given ps.png', bbox_inches = 'tight')
