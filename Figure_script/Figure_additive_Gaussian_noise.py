import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from matplotlib.lines import Line2D

# load traces
ts1 = pickle.load(open("../Data/45.pickle", "rb"))# no noise
ts2 = pickle.load(open("../Data/std=30%.pickle", "rb"))# additive noise: std=30%mean
ts = [ts1, ts2]
params = ['c', 'p50']
true_values = [16, -4.5]

# figure
labels = ['$\\mathit{c}$', '$\\psi_{x50}$ (MPa)']
ranges = [[2, 20], [-10, -0.1]]
custom_lines = [Line2D([0], [0], color='blue', lw=2),
                Line2D([0], [0], color='red', lw=2),
                Line2D([0], [0], linestyle='dashed', color='black', lw=2)]
fig, axs = plt.subplots(nrows=1, ncols=2, figsize=(20, 10))
for i, ax in enumerate(axs):
    param = params[i]
    df1 = pd.DataFrame({param: ts1[param]}).iloc[:, 0]
    kde1 = stats.gaussian_kde(df1)
    df2 = pd.DataFrame({param: ts2[param]}).iloc[:, 0]
    kde2 = stats.gaussian_kde(df2)
    mean1, std1 = df1.mean(), df1.std()
    mean2, std2 = df2.mean(), df2.std()
    cv1, cv2 = abs(std1/mean1), abs(std2/mean2)
    print(cv1, cv2, param)
    param_range = np.linspace(ranges[i][0], ranges[i][1], 1000)
    ax.plot(param_range, kde1(param_range), linewidth=2.5, color='blue')
    ax.plot(param_range, kde2(param_range), linewidth=2.5, color='red')
    ax.axvline(x=true_values[i], c='black', linestyle='dashed')
    ax.axes.get_yaxis().set_visible(False)
    ax.tick_params(labelsize=20)
    ax.set_xlabel(labels[i], fontsize=30)
    if i == 0:
        ax.legend(custom_lines, ['Without additive noise', 'With additive noise',
                                 'True value'],
                  loc='upper left', fontsize=20, framealpha=0)
plt.subplots_adjust(wspace=0.05)
plt.tight_layout
plt.show()
plt.savefig('../Figures/Figure additive Gaussian noise.png', bbox_inches = 'tight')
