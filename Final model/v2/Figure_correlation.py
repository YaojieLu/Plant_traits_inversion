import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#from sklearn.feature_selection import mutual_info_regression

# load MCMC output and thin
species_code = 'Aru_24'
species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}

ts = pickle.load(open('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                      .format(species_code), 'rb'))
ts['kxmax'] = 10**ts['kxmax_log10']
ts['alpha'] = 10**ts['alpha_log10']
latex = [r'$c$', '$g_1$', 'P50', r'$k_{xmax}/L$', r'$\alpha/L$']
df_thinned = {}
for key in ts:
    #print(np.median(ts[key]))
    df_thinned[key] = [item for index, item in enumerate(ts[key])\
                       if index % 10 == 0]
df_thinned = pd.DataFrame.from_dict(data=df_thinned, orient='columns')
df_thinned = df_thinned.drop(['kxmax_log10', 'alpha_log10'],\
                             axis=1)
print(df_thinned.columns)
df_thinned.rename(columns=dict(zip(df_thinned.columns, latex)),\
                  inplace=True)
#print(df_thinned[latex[-1]].describe(percentiles=[0.01]))

# figure
# correlation coefficient
def cf(x, y, **kwargs):
    # Calculate the value
    rho = np.corrcoef(x, y)
    # Make the label
    label = '$\\rho$ = ' + str(round(rho[0][1], 2))
    # Add the label to the plot
    ax = plt.gca()
    ax.annotate(label, xy=(0.15, 1.05), size=20, color='k',\
                xycoords=ax.transAxes)

# helper functions
def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)
sns.set_style('white')
#sns.set(font_scale=2)
sns.set_context('paper', rc={'xtick.labelsize': 12, 'ytick.labelsize': 12,\
                             'axes.labelsize': 20})
g = sns.PairGrid(data=df_thinned, vars=latex, diag_sharey=False,\
                 layout_pad=10)
g.fig.set_size_inches(12, 12)
g = g.map_lower(plt.scatter, s=1)
g = g.map_lower(cf)
g = g.map_diag(plt.hist)
#g = g.map_diag(sns.kdeplot, lw=3, legend=False)
#g = g.map_upper(plt.scatter, s=1)
g.map_upper(hide_current_axis)
plt.subplots_adjust(hspace=0.3, wspace=0.1)
g.savefig('../../Figures/Figure_correlation_Gil_v2_{}.png'\
          .format(species_code), bbox_inches="tight")
