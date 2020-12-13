
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# script input
species = 'Bpa'

# load MCMC output and thin
ts = pickle.load(open('../Data/UMB_trace/Bpa.pickle', 'rb'))
ts['alpha'] = 10**ts['alpha_log10']
ts['kxmax'] = 10**ts['kxmax_log10']

latex = [r'$b$', r'$c$', '$\\psi_{x50}$',\
         r'$k_{xmax} / g_1$', r'$k_{xmax} / \alpha$']
df_thinned = {}
for key in ts:
    #print(np.median(ts[key]))
    df_thinned[key] = [item for index, item in enumerate(ts[key])\
                       if index % 10 == 0]
df_thinned = pd.DataFrame.from_dict(data=df_thinned, orient='columns')
df_thinned['par1'] = df_thinned.kxmax/df_thinned.g1
df_thinned['par2'] = df_thinned.kxmax/df_thinned.alpha
df_thinned = df_thinned.drop(['alpha_log10', 'kxmax_log10', 'kxmax', 'g1',\
                              'alpha'], axis=1)
df_thinned.rename(columns=dict(zip(df_thinned.columns, latex)), inplace=True)

# figure
sns.set(font_scale=1)
g = sns.PairGrid(data=df_thinned, vars=latex, diag_sharey=False)
g.fig.set_size_inches(12, 6)
g = g.map_lower(plt.scatter, s=1)
g = g.map_diag(plt.hist)
#g = g.map_diag(sns.kdeplot, lw=3, legend=False)
#g = g.map_upper(plt.scatter, s=1)
def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
g.map_upper(hide_current_axis)
g.savefig('../Figures/Figure correlation Bpa.png', bbox_inches="tight")
