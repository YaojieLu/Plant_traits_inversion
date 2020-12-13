import os
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# load MCMC output and thin
species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}

# helper functions
def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

def f(species_code, plot=False, save=False):
    ts = pickle.load(open('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                          .format(species_code), 'rb'))
    ts['kxmax/g1'] = 10**ts['kxmax_log10']/10**ts['g1_log10']
    ts['kxmax/alpha'] = 10**ts['kxmax_log10']/10**ts['alpha_log10']
    latex = [r'$c$', '$\\psi_{x50}$', '$k_{xmax}/L/g_1$', r'$k_{xmax}/\alpha$']
    df_thinned = {}
    for key in ts:
        #print(np.median(ts[key]))
        df_thinned[key] = [item for index, item in enumerate(ts[key])\
                           if index % 10 == 0]
    df_thinned = pd.DataFrame.from_dict(data=df_thinned, orient='columns')
    df_thinned = df_thinned.drop(['g1_log10', 'kxmax_log10', 'alpha_log10'],\
                                 axis=1)
    df_thinned.rename(columns=dict(zip(df_thinned.columns, latex)),\
                      inplace=True)
    #print(df_thinned[latex[-1]].describe(percentiles=[0.01]))
    # figure
    if plot:
        #sns.set(font_scale=2)
        sns.set_context('paper', rc={'axes.labelsize': 24})
        plt.ioff()
        g = sns.PairGrid(data=df_thinned, vars=latex, diag_sharey=False)
        g.fig.set_size_inches(12, 12)
        g = g.map_lower(plt.scatter, s=1)
        g = g.map_diag(plt.hist)
        #g = g.map_diag(sns.kdeplot, lw=3, legend=False)
        #g = g.map_upper(plt.scatter, s=1)
        g.map_upper(hide_current_axis)
        if save:
            g.savefig('../../Figures/Figure_correlation_Gil_v2_{}.png'\
                      .format(species_code), bbox_inches="tight")

# figures
#species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
species_codes = ['Aru_24']
for species_code in species_codes:
    if not os.path.exists('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                          .format(species_code)):
        continue
    f(species_code, plot=True, save=True)
    print(species_code)
