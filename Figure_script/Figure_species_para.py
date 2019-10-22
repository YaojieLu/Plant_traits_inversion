import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# load traces
Am = pickle.load(open("../Data/Am.pickle", "rb"))
Pm = pickle.load(open("../Data/Pm.pickle", "rb"))
df_ts = pd.DataFrame({'p50': {'Am': Am['p50'], 'Pm': Pm['p50']},
                      'c': {'Am': Am['c'], 'Pm': Pm['c']},
                      'Z': {'Am': Am['Z'], 'Pm': Pm['Z']}})

# figure
labels = ['Pacific Madrone', 'Douglas fir']
paras = ['Xylem vulnerability (MPa)', 'Stomatal sensitivity', 'Rooting depth (m)']
fig = plt.figure(figsize = (24, 12))
# parameter histograms
for i in range(len(paras)):
    ax = fig.add_subplot(2, 3, i+1)
    ts_Am, ts_Pm = df_ts.iloc[:, i]
    x_min = np.floor(min(ts_Am.min(), ts_Pm.min()))
    x_max = np.ceil(max(ts_Am.max(), ts_Pm.max()))
    x = np.linspace(x_min, x_max, 1000)
    kde_Am = stats.gaussian_kde(ts_Am)(x)
    kde_Pm = stats.gaussian_kde(ts_Pm)(x)
    ax.hist(ts_Am, color = 'k', density = True, bins = 30, alpha = 0.3)
    #ax.plot(x, kde_Am, color = 'k')
    ax.hist(ts_Pm, color = 'b', density = True, bins = 30, alpha = 0.3)
    #ax.plot(x, kde_Pm, color = 'b')
    plt.xlim([x_min, x_max])
    plt.xlabel(paras[i], fontsize = 20)
    plt.tick_params(axis = 'x', labelsize = 20)
    plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
    plt.tight_layout
plt.legend(labels, loc = 'upper left', fontsize = 20, framealpha = 0)
# parameter difference histograms
for i in range(len(paras)):
    ax = fig.add_subplot(2, 3, i+4)
    ts_Am, ts_Pm = df_ts.iloc[:, i]
    ts = ts_Am - ts_Pm
    neg = 100*np.sum(ts <= 0, axis = 0)/len(ts)
    x_min = np.floor(min(ts))
    x_max = np.ceil(max(ts))
    x = np.linspace(x_min, x_max, 1000)
    ax.hist(ts, density = True, bins = 30, alpha = 0.3)
    plt.axvline(x = 0, color = 'r', linestyle = '--')
    y_pos = ax.get_ylim()[1]/2
    ax.annotate('{}%'.format(int(neg)),
                xy = (x_min*0.3, y_pos*0.05),# xycoords = 'axes fraction',
                fontsize = 20, color = 'r')
    ax.annotate('{}%'.format(int(100-neg)),
                xy = (x_max*0.08, y_pos*0.05),# xycoords = 'axes fraction',
                fontsize = 20, color = 'r')
    plt.xlim([x_min, x_max])
    plt.xlabel(paras[i], fontsize = 20)
    plt.tick_params(axis = 'x', labelsize = 20)
    plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
    plt.tight_layout
    if i == 1:
        plt.title("The difference between Pacific Madrone and Douglas fir", fontsize = 20)
plt.subplots_adjust(wspace = 0.1, hspace = 0.35)
plt.savefig('../Figures/Figure species parameters.png', bbox_inches = 'tight')
