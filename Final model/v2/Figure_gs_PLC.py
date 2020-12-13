import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.lines import Line2D

species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
species_codes.remove('Pst_1')
species_codes.remove('Pst_19')
for species_code in species_codes:
    if not os.path.exists('../../Results/test/ensemble_gs_PLC_UMB_Gil_v2_2_{}.csv'\
                          .format(species_code)):
        species_codes.remove(species_code)
species_codes.sort()

# figure
species_abbrs = ['Aru', 'Bpa', 'Pgr', 'Pst']
species_names = ['Red maple', 'Paper birch', 'Bigtooth aspen',\
                 'White pine']
species_names = dict(zip(species_abbrs, species_names))
colors = sns.color_palette()[:4]
colors = dict(zip(species_abbrs, colors))
fig, axs = plt.subplots(nrows=5, ncols=5, figsize=(30, 20), sharex=True,\
                        sharey=True)
i = 0
for row in axs:
    for col in row:
        if i < len(species_codes):
            species_code = species_codes[i]
            i += 1
            loc = '../../Results/test/ensemble_gs_PLC_UMB_Gil_v2_2_{}.csv'\
                .format(species_code)
            df = pd.read_csv(loc)
            df = df.dropna()
            df['gs'] = df['gs'].apply(lambda x: 0.05*round(x/0.0005))
            df_5 = df.groupby('gs')['PLC'].quantile(.05)
            df_50 = df.groupby('gs')['PLC'].median()
            df_95 = df.groupby('gs')['PLC'].quantile(.95)
            df = pd.concat([df_5, df_50, df_95], axis=1, join='inner')
            df.reset_index(inplace=True)
            df.columns = ['gs', '5', '50', '95']
            col.plot(df['gs'], df['50']*100, color=colors[species_code[:3]])
            col.fill_between(df['gs'], df['5']*100, df['95']*100, color='b',\
                             alpha=0.2)
            col.axvline(x=10, linestyle='-')
        else:
            i += 1
            col.plot(df['gs'], df['50']*100, lw=0)
        col.set_xlim([0, 100])
        col.set_ylim([0, 100])
        col.tick_params(labelsize=30)
fig.text(0.06, 0.5, 'Percentage loss of conductivity (%)', va='center',\
         rotation='vertical', fontsize=50)
fig.text(0.31, 0.05, 'Normalized stomatal conductance (%)', va='center',\
         fontsize=50)
custom_lines = [Line2D([0], [0], lw=4,\
                       color=colors[s], label=species_names[s])\
                for s in species_abbrs]
fig.legend(handles=custom_lines, prop={'size': 50}, loc='upper center',\
           ncol=len(species_abbrs), frameon=False)
fig.subplots_adjust(top=0.89)
fig.savefig('../../Figures/ensemble_gs_PLC.png', bbox_inches='tight')
