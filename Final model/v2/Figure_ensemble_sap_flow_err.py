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
fig, axs = plt.subplots(nrows=4, ncols=5, figsize=(30, 20), sharex=True)
i = 0
for row in axs:
    for col in row:
        if i < len(species_codes):
            species_code = species_codes[i]
            i += 1
            df = pd.read_csv('../../Results/ensemble_vn_UMB_Gil_v2_{}.csv'\
                             .format(species_code))
            df['observed'] = df['observed'].replace({0: np.nan})
            #df = df.dropna()
            vn_max = df['observed'].max()
            df['observed'] = df['observed']/vn_max
            df['qt=0.5'] = df['qt=0.5']/vn_max
            col.scatter(df['date'], df['observed']-df['qt=0.5'],\
                     color=colors[species_code[:3]])
            std = 0.1*df['observed'].mean()
            col.plot(df['date'], df['observed']*0.1)
            col.plot(df['date'], -df['observed']*0.1)
            col.set_title(r'{}'.format(species_code))
            col.xaxis.set_major_locator(ticker.MultipleLocator(50))
            col.xaxis.set_tick_params(rotation=40)
plt.subplots_adjust(hspace=0.2, wspace=0.1)
