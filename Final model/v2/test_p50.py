import pickle
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

# observed
Aru = [-3.028333333, -3.028333333, -3.794854811, -3.3, -1.97, -1.97, -1.97,\
       -1.69, -1.641892351, ]
Bpa = [-2.1685, -2.1685, -2.337, -2.337, -2.337, -2.14550845, -1.596]
Pgr = [-1.52]
obs = pd.DataFrame({'species': ['Aru']*len(Aru)+['Bpa']*len(Bpa)+\
                    ['Pgr']*len(Pgr), 'p50': Aru+Bpa+Pgr})

# inferred
species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
species_name_dict = {'Aru': ['Red maple', 'Acer rubrum'],
                     'Bpa': ['Paper birch', 'Betula papyrifera'],
                     'Pgr': ['Bigtooth aspen', 'Populus grandidentata'],
                     'Pst': ['White pine', 'Pinus strobus']}

df = None
for species_code in species_codes:
    if not os.path.exists('../../Data/UMB_trace/Gil_v2/test/{}.pickle'\
                          .format(species_code)):
        continue
    ts = pickle.load(open('../../Data/UMB_trace/Gil_v2/test/{}.pickle'\
                          .format(species_code), 'rb'))['p50']
    temp = pd.DataFrame(ts, columns=['p50'])
    temp['species_code'] = species_code
    temp['Species'] = species_name_dict[species_code[:3]][0]
    if df is None:
        df = temp
    else:
        df = pd.concat([df, temp])
df = df.sort_values(by=['species_code'])

# figure
sns.set_style('dark')
sns.set(font_scale=2)
fig, axs = plt.subplots(1, 2, gridspec_kw={'width_ratios':[3, 1]},\
                        figsize=(80/3, 10), sharey=True)
sns.violinplot(x='species_code', y='p50', hue='Species', data=df,\
               dodge=False, ax=axs[0])
axs[0].set(xticklabels=[])
axs[0].set(xlabel=None)
axs[0].tick_params(bottom=False)
axs[0].set_ylabel(ylabel='P50 (MPa)')
axs[0].set(ylim=(-8, 0))
axs[0].legend(title_fontsize=25, bbox_to_anchor=(1.07, 1.1), ncol=4)
axs[0].text(-0.2, -0.2, 'a) inferred')
sns.scatterplot(x='species', y='p50', hue='species', data=obs, s=99,\
                legend=False, ax=axs[1])
axs[1].set(xticklabels=[])
axs[1].set(xlabel=None)
axs[1].tick_params(bottom=False)
axs[1].set(ylim=(-10, 0))
axs[1].xaxis.grid(False)
axs[1].text(0.01, -0.2, 'b) literature')
fig.subplots_adjust(wspace=0.02)
