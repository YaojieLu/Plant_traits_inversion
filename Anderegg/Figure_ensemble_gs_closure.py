
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# species information
species_name_dict = {'Aru': ['Red maple', 'Acer rubrum'],
                'Bpa': ['Paper birch', 'Betula papyrifera'],
                'Pgr': ['Bigtooth aspen', 'Populus grandidentata'],
                'Pst': ['Eastern white pine', 'Pinus strobus']}

# figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
species_list = ['Aru', 'Bpa', 'Pgr', 'Pst']
i = 0
for row in axs:
    for col in row:
        sp = species_list[i]
        df = pd.read_csv('../Results/ensemble_gs_UMB_{}.csv'.format(sp))
        df.sort_values(by=['ps'], inplace=True)
        col.scatter(df['ps'], df['qt=0.5'], color='k')
        col.fill_between(df['ps'], df['qt=0.05'], df['qt=0.95'],\
                         color='b', alpha=0.2)
        label = species_name_dict[list(species_name_dict.keys())[i]][0]
        col.set_title(r'{}, {}'.format(label, sp), fontsize=30)
        col.set_ylim(-0.05, 1.05)
        col.tick_params(labelsize=20)
        #xlabels = []
        if i > 1:
            col.set_xlabel('$\\psi_{x}$ (MPa)', fontsize=30)
        else:
            col.axes.get_xaxis().set_visible(False)
        if i == 0 or i == 2:
            col.set_ylabel('Stomatal response', fontsize=30)
        else:
            col.axes.get_yaxis().set_visible(False)
        i += 1
        print(df['qt=0.95'].min())
plt.subplots_adjust(hspace=0.2, wspace=0.1)
