import os
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec
import seaborn as sns

species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
species_name_dict = {'Aru': ['Red maple', 'Acer rubrum'],
                     'Bpa': ['Paper birch', 'Betula papyrifera'],
                     'Pgr': ['Bigtooth aspen', 'Populus grandidentata'],
                     'Pst': ['Eastern white pine', 'Pinus strobus']}

df = None
for species_code in species_codes:
    if not os.path.exists('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                          .format(species_code)):
        continue
    ts = pickle.load(open('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                          .format(species_code), 'rb'))['p50']
    temp = pd.DataFrame(ts, columns=['p50'])
    temp['species_code'] = species_code
    temp['Species'] = species_name_dict[species_code[:3]][0]
    if df is None:
        df = temp
    else:
        df = pd.concat([df, temp])
df = df.sort_values(by=['species_code'])
df = df[~df['species_code'].isin(['Bpa_31', 'Pgr_27', 'Pst_19'])]

# figure
colors = dict(zip(['Aru', 'Bpa', 'Pgr', 'Pst'], sns.color_palette()[:4]))
species_codes = np.unique(df['species_code'])
gs = (grid_spec.GridSpec(len(species_codes), 1))
fig = plt.figure(figsize=(8, 6))
i = 0
ax_objs = []
for species_code in species_codes:
    ax_objs.append(fig.add_subplot(gs[i: i+1, 0:]))

    plot = (df[df['species_code'] == species_code]['p50']\
            .plot.kde(ax=ax_objs[-1],color="#f0f0f0", lw=0.5))

    x = plot.get_children()[0]._x
    y = plot.get_children()[0]._y
    ax_objs[-1].fill_between(x, y, color=colors[species_code[:3]])
    ax_objs[-1].set_xlim(-3.5, 0)
    ax_objs[-1].xaxis.set_ticks_position('none')
    ax_objs[-1].axes.get_yaxis().set_visible(False)
    i += 1
    
    rect = ax_objs[-1].patch
    rect.set_alpha(0)
    if i == len(species_codes)-1:
        pass
    else:
        ax_objs[-1].set_xticklabels([])
    spines = ['top', 'right', 'left', 'bottom']
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)
gs.update(hspace= -0.5)
