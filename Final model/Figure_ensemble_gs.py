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
        df = pd.read_csv('../Results/ensemble_gs_UMB_{}.csv'.format(sp)).iloc[:, :]
        df = df[df['qt=0.95'] <= 1]
        col.plot(df['date'], df['qt=0.5'], color='k')
        col.fill_between(df['date'], df['qt=0.05'], df['qt=0.95'],
                        color='b', alpha=0.2)
        label = species_name_dict[list(species_name_dict.keys())[i]][0]
        col.set_title(r'{}, {}'.format(label, sp), fontsize=30)
        col.tick_params(labelsize=20)
        if i > 1:
            col.xaxis.set_major_locator(ticker.MultipleLocator(20))
            col.xaxis.set_tick_params(rotation=20)
        else:
            col.axes.get_xaxis().set_visible(False)
        if i == 0 or i == 2:
            col.set_ylabel(r'$\it{g_s}\ \mathrm{(mol\ m^{-2}\ s^{-1})}$',\
                           fontsize=30)
        else:
            col.axes.get_yaxis().set_visible(False)
        i += 1
plt.subplots_adjust(hspace=0.2, wspace=0.05)
