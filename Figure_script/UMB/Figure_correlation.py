
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import mutual_info_regression
from scipy.stats import pearsonr

# load MCMC output and thin
species = 'Aru'
ts = pickle.load(open('../../Data/UMB_trace/MCMC.pickle', 'rb'))
latex = ['$\\alpha$', r'$c$', '$\\psi_{x50}$',
         r'$k_{xmax}$', r'$L \times g_1$']
df_thinned = {}
for key in ts:
    df_thinned[key] = [item for index, item in enumerate(ts[key])]# if index % 50 == 0]
df_thinned = pd.DataFrame.from_dict(data=df_thinned, orient='columns')
df_thinned['par1'] = df_thinned.L*df_thinned.g1
df_thinned = df_thinned.drop(['L', 'g1'], axis=1)
df_thinned.rename(columns=dict(zip(df_thinned.columns, latex)), inplace=True)

# mutual information
def mif(x, y, **kwargs):
    x = x.values.reshape(-1, 1)
    # Calculate the value
    mi = mutual_info_regression(x, y, discrete_features=False)
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
    ax.annotate(label, xy=(0.15, 1.02), size=20, color=c, xycoords=ax.transAxes)

# pearson correlation coefficient
def pcf(x, y, **kwargs):
    # Calculate the value
    (r, p) = pearsonr(x, y)
    # Make the label
    label = '$\\rho$ = ' + str(round(r, 2))
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy=(0.15, 1.02), size=20, xycoords=ax.transAxes)

# figure
sns.set(font_scale=1.8)
g = sns.PairGrid(data=df_thinned, vars=latex, diag_sharey=False)
#g = g.map_lower(sns.kdeplot, cmap='Blues_d')
g = g.map_lower(plt.scatter, s=1)
g = g.map_lower(pcf)
g = g.map_diag(plt.hist)
#g = g.map_diag(sns.kdeplot, lw=3, legend=False)
#g = g.map_upper(plt.scatter, s=1)
def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
g.map_upper(hide_current_axis)
plt.savefig('../../Figures/Figure UMB correlation {}.png'.format(species), bbox_inches='tight')
