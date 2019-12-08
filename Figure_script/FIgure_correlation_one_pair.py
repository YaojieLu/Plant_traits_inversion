
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# load MCMC output and thin
ts = pickle.load(open("../Data/45.pickle", "rb"))
latex = ['$\\alpha$', r'$c$', '$\\psi_{x50}$',
         r'$k_{xmax}$', r'$L \times g_1$']
df_thinned = {}
for key in ts:
    df_thinned[key] = [item for index, item in enumerate(ts[key])]# if index % 50 == 0]
df_thinned = pd.DataFrame.from_dict(data=df_thinned, orient='columns')
df_thinned['par1'] = df_thinned.L*df_thinned.g1
df_thinned = df_thinned.drop(['L', 'g1'], axis=1)
df_thinned.rename(columns=dict(zip(df_thinned.columns, latex)), inplace=True)

# figure
sns.set(font_scale=1.8)
g = sns.PairGrid(data=df_thinned, vars=latex, diag_sharey=False)
g = g.map_lower(sns.kdeplot, cmap="Blues_d")
g = g.map_diag(plt.hist)
#g = g.map_diag(sns.kdeplot, lw=3, legend=False)
g = g.map_upper(plt.scatter, s=1)
plt.savefig('../Figures/Figure correlation.png', bbox_inches='tight')
