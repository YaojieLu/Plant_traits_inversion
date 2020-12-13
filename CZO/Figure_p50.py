import pickle
import matplotlib.pyplot as plt

# tracs
species_name_dict = {'Aru': ['Red maple', 'Acer rubrum'],
                     'Bpa': ['Paper birch', 'Betula papyrifera'],
                     'Pgr': ['Bigtooth aspen', 'Populus grandidentata'],
                     'Pst': ['Eastern white pine', 'Pinus strobus']}
traces = []
for sp in species_name_dict.keys():
    traces.append(pickle.load(open('../Data/UMB_trace/{}.pickle'.format(sp),\
                                   'rb'))['p50'])
traces_thinned = []
for t in traces:
    traces_thinned.append([item for index, item in enumerate(t)\
                           if index % 10 == 0])

# figure
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 15))
for i, s in enumerate(list(species_name_dict.keys())):
    label = species_name_dict[list(species_name_dict.keys())[i]][0]
    ax.hist(traces_thinned[i], label=label, density=True, bins=20)
    ax.set_xlim([-2.4, -0.7])
    ax.axes.get_yaxis().set_visible(False)
plt.xlabel('$\\psi_{x50}$ (MPa)', fontsize=30)
plt.tick_params('both', labelsize=30)
plt.legend(fontsize=30)
plt.tight_layout
plt.savefig('../Figures/Figure p50.png', bbox_inches='tight')
