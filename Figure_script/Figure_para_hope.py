import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# load traces
ts_30 = pickle.load(open("../Data/30.pickle", "rb"))
ts_45 = pickle.load(open("../Data/45.pickle", "rb"))
ts_60 = pickle.load(open("../Data/60.pickle", "rb"))
ts = [ts_30, ts_45, ts_60]
params = ['c', 'p50']
true_values = [[16, -3], [16, -4.5], [16, -6]]

# figure
labels = ['$\\mathit{c}$', '$\\psi_{x50}$ (MPa)']
ranges = [[2, 20], [-10, -0.1]]
fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(20, 30))
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        param = params[j]
        df = pd.DataFrame({param: ts[i][param]}).iloc[:, 0]
        kde = stats.gaussian_kde(df)
        param_range = np.linspace(ranges[j][0], ranges[j][1], 1000)
        col.plot(param_range, kde(param_range), linewidth=2.5, color='blue')
        mean, std = round(df.mean(), 2), round(df.std(), 2)
        cv = abs(round(std/mean, 2))
        col.set_title('Mean = {}; relative Std = {}'.format(mean, cv), fontsize=20)
        col.axvline(x=true_values[i][j], c='black',
                    label='True value', linestyle='dashed')
        if i!=2:
            col.axes.get_xaxis().set_visible(False)
        col.axes.get_yaxis().set_visible(False)
        col.tick_params(labelsize=20)
        if i==2:
            col.set_xlabel(labels[j], fontsize=30)
        if (i==0 and j==0):
            col.legend(loc='upper left', fontsize=20, framealpha=0)

plt.subplots_adjust(hspace=0.08, wspace=0.05)
plt.tight_layout
plt.show()
plt.savefig('../Figures/Figure posterior distributions.png', bbox_inches = 'tight')
