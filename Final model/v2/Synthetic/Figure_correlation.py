import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

species_codes = ['vn_0_ps_0', 'vn_0_ps_05', 'vn_05_ps_0', 'vn_05_ps_05']

# helper functions
def hide_current_axis(*args, **kwds):
    plt.gca().set_visible(False)

def f(species_code, plot=False, save=False):
    ts = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                          .format(species_code), 'rb'))
    print('c: {};\np50: {}\nkxmax_log10: {}\nalpha_log10: {}\ng1_log10: {}'\
          .format(ts['c'].mean(), ts['p50'].mean(), ts['kxmax_log10'].mean(),\
                  ts['alpha_log10'].mean(), ts['g1_log10'].mean()))
    latex = [r'$c$', '$\\psi_{x50}$', '$\log(k_{xmax}/L)$',\
             r'$\log(\alpha)$', '$\log(g_1)$']
    df_thinned = {}
    for key in ts:
        #print(np.median(ts[key]))
        df_thinned[key] = [item for index, item in enumerate(ts[key])\
                           if index % 10 == 0]
    df_thinned = pd.DataFrame.from_dict(data=df_thinned, orient='columns')
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
            g.savefig('../../../Figures/Figure_correlation_Gil_v2_{}.png'\
                      .format(species_code), bbox_inches="tight")

# figures
for species_code in species_codes:
    print(species_code)
    f(species_code, plot=True, save=True)
