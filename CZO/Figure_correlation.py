import pickle
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

# script input
species = 'Nd'

# load MCMC output and thin
ts = pickle.load(open('../Data/CZO_trace/{}.pickle'.format(species), 'rb'))
ts['alpha'] = 10**ts['alpha_log10']
ts['c'] = 10**ts['c_log10']
ts['g1'] = 10**ts['g1_log10']
ts['kxmax'] = 10**ts['kxmax_log10']

latex = ['$\\psi_{x50}$', '$\\psi_{s}$', r'$\alpha$', 
         r'$c$', r'$g_{1}$', r'$k_{xmax}$']
df_thinned = {}
for key in ts:
    #print(np.median(ts[key]))
    df_thinned[key] = [item for index, item in enumerate(ts[key])\
                       if index % 1 == 0]
df_thinned = pd.DataFrame.from_dict(data=df_thinned, orient='columns')
df_thinned = df_thinned.drop(['alpha_log10', 'c_log10', 'g1_log10',\
                              'kxmax_log10'], axis=1)
df_thinned = df_thinned[(df_thinned['alpha']<20) & (df_thinned['c']<2) &\
                        (df_thinned['g1']<100) &\
                        (df_thinned['kxmax']<0.2*10**8)]
df_thinned.rename(columns=dict(zip(df_thinned.columns, latex)), inplace=True)

# figure
#sns.set(font_scale=2)
sns.set_context('paper', rc={'axes.labelsize': 24})
g = sns.PairGrid(data=df_thinned, vars=latex, diag_sharey=False)
g.fig.set_size_inches(12, 12)
g = g.map_lower(plt.scatter, s=1)
g = g.map_diag(plt.hist)
#g = g.map_diag(sns.kdeplot, lw=3, legend=False)
#g = g.map_upper(plt.scatter, s=1)
def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
g.map_upper(hide_current_axis)
plt.tight_layout()
g.savefig('../Figures/Figure correlation {}.png'.format(species),\
          bbox_inches='tight')
