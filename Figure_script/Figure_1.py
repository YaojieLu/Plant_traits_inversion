import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

# load traces
ts = pickle.load(open("../Data/45.pickle", "rb"))
params = ['alpha', 'c', 'g1', 'kxmax', 'p50', 'L']
true_values = [0.02, 16, 50, 7, -4.5, 2]

# figure
labels = ['$\\alpha$', '$\\mathit{c}$', '$\\mathit{g_{1}}$',
          '$\\mathit{k_{xmax}}$', '$\\psi_{x50}$', '$\\mathit{L}$']
ranges = [[0.001, 0.2], [2, 20], [10, 100], [1, 10], [-10, -0.1], [0.5, 5]]
fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(30, 20))
for i, row in enumerate(axs):
    for j, col in enumerate(row):
        idx = i*3+j
        param = params[idx]
        df = pd.DataFrame({param: ts[param]}).iloc[:, 0]
        col.hist(df, range=[ranges[idx][0], ranges[idx][1]], bins=100)
#        kde = stats.gaussian_kde(df)
#        param_range = np.linspace(ranges[idx][0], ranges[idx][1], 1000)
#        col.plot(param_range, kde(param_range), linewidth=2.5, color='blue')
        mean, std = df.mean(), df.std()
        cv = abs(round(std/mean, 2))
        col.set_title('RSD = {}'.format(cv), fontsize=30)
        col.axvline(x=true_values[idx], c='black',
                    label='True value', linestyle='dashed')
        col.axes.get_yaxis().set_visible(False)
        col.tick_params(labelsize=30)
        col.set_xlabel(labels[idx], fontsize=30)
        if idx == 0:
            col.legend([Line2D([0], [0], linestyle='dashed', color='black')],
                       ['True value'], loc='upper right', fontsize=30, framealpha=0)

plt.subplots_adjust(hspace=0.25, wspace=0.1)
plt.savefig('../Figures/Figure 45.png', bbox_inches = 'tight')
