import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.lines import Line2D

species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
for species_code in species_codes:
    if not os.path.exists('../../Results/ensemble_vn_UMB_Gil_v2_{}.csv'\
                          .format(species_code)):
        species_codes.remove(species_code)
species_codes.sort()

# figure
species_abbrs = ['Aru', 'Bpa', 'Pgr', 'Pst']
species_names = ['Red maple', 'Paper birch', 'Bigtooth aspen',\
                 'Eastern white pine']
species_names = dict(zip(species_abbrs, species_names))
colors = sns.color_palette()[:4]
colors = dict(zip(species_abbrs, colors))
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(30, 20), sharex=True,\
                        sharey=True)
i = 0
for row in axs:
    for col in row:
        if i < len(species_codes):
            species_code = species_codes[i]
            i += 1
            df = pd.read_csv('../../Results/ensemble_PLC_UMB_Gil_v2_{}.csv'\
                             .format(species_code))
            columns = ['qt=0.05', 'qt=0.5', 'qt=0.95']
            #df = df.dropna()
            col.plot(df['date'], df['qt=0.5'], color=colors[species_code[:3]])
            col.fill_between(df['date'], df['qt=0.05'], df['qt=0.95'],\
                             color='b', alpha=0.2)
            col.set_title(r'{}'.format(species_code), fontsize=30)
            col.xaxis.set_major_locator(ticker.MultipleLocator(50))
            col.xaxis.set_tick_params(rotation=40)
            col.tick_params(labelsize=30)
plt.subplots_adjust(hspace=0.2, wspace=0.1)
fig.text(0.09, 0.5, 'PLC',\
         va='center', rotation='vertical', fontsize=30)
custom_lines = [Line2D([0], [0], lw=4, color='black', label='Data')]+\
               [Line2D([0], [0], lw=4,\
                       color=colors[s], label=species_names[s])\
                for s in species_abbrs]
fig.legend(handles=custom_lines, prop={'size': 30}, loc='upper center',\
          ncol=len(species_abbrs)+1, frameon=False)
fig.subplots_adjust(top=0.9)
#fig.savefig('../../Figures/ensemble_sap_flow.png', bbox_inches='tight')
