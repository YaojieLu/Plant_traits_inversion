
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression

# script input
species, run = 'Aru', 15

# load MCMC output and thin
ts = pickle.load(open('../Data/UMB_trace/{}.pickle'.format(species), 'rb'))
latex = ['$\\alpha$', '$\\mathit{c}$', '$\\psi_{x50}$',
         '$\\mathit{k_{xmax}}$', '$\\mathit{g_1}$']
df_thinned = {}
for key in ts:
    df_thinned[key] = [item for index, item in enumerate(ts[key]) if index % 50 == 0]
df_thinned = pd.DataFrame.from_dict(data = df_thinned, orient = 'columns')
df_thinned.rename(columns = dict(zip(ts.keys(), latex)), inplace = True)

# mutual information
def mif(x, y, **kwargs):
    x = x.values.reshape(-1, 1)
    # Calculate the value
    mi = mutual_info_regression(x, y, discrete_features = False)
    # Make the label
    label = 'MI = ' + str(round(mi[0], 2))
    # Add the label to the plot
    ax = plt.gca()
    if mi > 0.2:
        c = 'r'
    elif 0.2 >= mi > 0.1:
        c = 'k'
    else:
        c = 'lightgrey'
    #c = 'r' if mi > 0.1 else 'lightgrey'
    ax.annotate(label, xy = (0.15, 1.02), size = 20, color = c, xycoords = ax.transAxes)

# figure
sns.set(font_scale = 1.8)
g = sns.PairGrid(data = df_thinned, vars = latex, diag_sharey = False)
g = g.map_lower(plt.scatter, s = 1)
g = g.map_lower(mif)
g = g.map_diag(plt.hist)
def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
g.map_upper(hide_current_axis)
#g.axes[0, 0].set_xlim(0, 0.06)
#g.axes[0, 6].set_xlim(0, 0.003)
#g.axes[6, 0].set_ylim(0, 0.003)
#plt.savefig('../Figures/Figure correlation.png', bbox_inches = 'tight')
