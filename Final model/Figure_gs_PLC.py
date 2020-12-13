import pandas as pd
import matplotlib.pyplot as plt

# species information
species_name_dict = {'Aru': ['Red maple', 'Acer rubrum'],
                     'Bpa': ['Paper birch', 'Betula papyrifera'],
                     'Pgr': ['Bigtooth aspen', 'Populus grandidentata'],
                     'Pst': ['Eastern white pine', 'Pinus strobus']}

# figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
i = 0
for row in axs:
    for col in row:
        sp = list(species_name_dict.keys())[i]
        PLC = pd.read_csv('../Results/ensemble_PLC_UMB_{}.csv'.format(sp))
        gs = pd.read_csv('../Results/ensemble_gs_UMB_{}.csv'.format(sp))
        col.scatter(PLC['qt=0.5'], gs['qt=0.5'], color='k')
        label = species_name_dict[list(species_name_dict.keys())[i]][0]
        col.set_title(r'{}, {}'.format(label, sp), fontsize=30)
        col.tick_params(labelsize=20)
        col.set_xlim([0, 1])
        col.set_ylim([0, 1])
        if i > 1:
            col.set_xlabel('PLC', fontsize=30)
        else:
            col.axes.get_xaxis().set_visible(False)
        if i == 0 or i == 2:
            col.set_ylabel('relative gs', fontsize=30)
        else:
            col.axes.get_yaxis().set_visible(False)
        i += 1
plt.subplots_adjust(hspace=0.2, wspace=0.1)
