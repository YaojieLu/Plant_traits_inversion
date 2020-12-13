import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# species information
species_name_dict = {'Am': ['Pacific madrone', 'Arbutus menziesii'],
                     'Nd': ['Tanoak', 'Notholithocarpus densiﬂorus'],
                     'Pm': ['Douglas fir', 'Pseudotsuga menziesii'],
                     'Qc': ['Oregon white oak', 'Quercus garryana'],
                     'Qg': ['Canyon live oak', 'Quercus chrysolepis']}

# figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
species_list = ['Am', 'Nd', 'Pm', 'Qg']
i = 0
for row in axs:
    for col in row:
        sp = species_list[i]
        df = pd.read_csv('../Results/ensemble_vn_CZO_{}.csv'.format(sp))\
            #.iloc[:60, :]
        df = df[df['qt=0.95'] <= 1]
        col.plot(pd.to_datetime(df['date']), df['qt=0.5'], color='k')
        col.plot(pd.to_datetime(df['date']), df['observed'],\
                 color='lightcoral')
        col.fill_between(pd.to_datetime(df['date']), df['qt=0.05'],\
                         df['qt=0.95'], color='b', alpha=0.2)
        label = species_name_dict[list(species_name_dict.keys())[i]][0]
        r2 = df['qt=0.5'].corr(df['observed'])
        col.set_title(r'{}, {}, $R^{}$ = {:0.2f}'.format(label, sp, 2, r2),\
                      fontsize=30)
        col.tick_params(labelsize=20)
        if i > 1:
            col.xaxis.set_major_locator(ticker.MultipleLocator(20))
            col.xaxis.set_tick_params(rotation=20)
        else:
            col.axes.get_xaxis().set_visible(False)
        col.set_ylabel('Sap velocity', fontsize=30)
        col.set_ylim([0, 1])
        if i == 0:
            col.legend(['model', 'data'], loc='upper left', fontsize=20,\
                       framealpha=0)
        i += 1
plt.subplots_adjust(hspace=0.2, wspace=0.2)
