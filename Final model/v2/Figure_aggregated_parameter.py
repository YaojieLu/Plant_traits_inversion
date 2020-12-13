import pickle
import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt

species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
species_codes.remove('Pst_1')
species_codes.remove('Pst_19')
species_name_dict = {'Aru': ['Red maple', 'Acer rubrum'],
                     'Bpa': ['Paper birch', 'Betula papyrifera'],
                     'Pgr': ['Bigtooth aspen', 'Populus grandidentata'],
                     'Pst': ['White pine', 'Pinus strobus']}

df = None
for species_code in species_codes:
    if not os.path.exists('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                          .format(species_code)):
        continue
    ts = pickle.load(open('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                          .format(species_code), 'rb'))
    param1 = 10**ts['kxmax_log10']/ts['g1']
    param2 = 10**ts['kxmax_log10']/10**ts['alpha_log10']
    temp = pd.DataFrame(list(zip(param1, param2)),\
                        columns=['kxmax/g1', 'kxmax/alpha'])
    temp['species_code'] = species_code
    temp['Species'] = species_name_dict[species_code[:3]][0]
    if df is None:
        df = temp
    else:
        df = pd.concat([df, temp])
df = df.sort_values(by=['species_code'])
#df = df[~df['species_code'].isin(['Bpa_31', 'Pgr_27', 'Pst_19'])]

# figure
sns.set(font_scale=2.5)
sns.set_style('white')
fig, axs = plt.subplots(2, figsize=(20, 20))
sns.violinplot(x='species_code', y='kxmax/alpha', hue='Species', data=df,\
               dodge=False, ax=axs[0])
axs[0].set(xticklabels=[])
axs[0].set(xlabel=None)
axs[0].tick_params(bottom=False)
axs[0].set_ylabel(ylabel=r'$k_{xmax}/\alpha$'\
                         '\n'\
                         r'$(\mathrm{mmol\ xylem\ m^{-2}\ s^{-1}\ MPa^{-1}})$',\
                         fontsize=30)
axs[0].set(ylim=(0, 1e4))
axs[0].legend(bbox_to_anchor=(0.5, 1.13), ncol=4, loc='upper center')
sns.violinplot(x='species_code', y='kxmax/g1', hue='Species', data=df,\
               dodge=False, ax=axs[1])
axs[1].set(xticklabels=[])
axs[1].set(xlabel=None)
axs[1].tick_params(bottom=False)
axs[1].set_ylabel(ylabel='$k_{xmax}/L/g_1$'\
                         '\n'\
                         r'$(\mathrm{mmol\ ground\ m^{-2}\ s^{-1}\ MPa^{-1}})$'\
                         '\n',\
                         fontsize=30)
axs[1].set(ylim=(0, 30))
axs[1].legend().set_visible(False)
fig.subplots_adjust(hspace=0.06)
fig.savefig('../../Figures/aggregated_parameters.png', bbox_inches='tight')
