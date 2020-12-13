import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

labels = ['baseline', 'small', 'large']
df = None
for label in labels:
    ts = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                          .format(label), 'rb'))['p50']
    temp = pd.DataFrame(ts, columns=['p50'])
    temp['label'] = label
    if df is None:
        df = temp
    else:
        df = pd.concat([df, temp])

# figure
names = ['Baseline', 'Small noise', 'Large noise']
names = dict(zip(labels, names))
colors = ['#3300cc', '#990066', '#ff0000']
colors = dict(zip(labels, colors))
fig, ax = plt.subplots(1, 1, figsize=(8, 6))
for label in labels:
    df[df['label']==label]['p50']\
        .hist(ax=ax, bins=500, color=colors[label],\
              histtype='step', density=True)
true_p50 = -2.5
ax.axvline(x=true_p50, linestyle='--', color='black')
ax.grid(False)
ax.set_xlim([-5, 0])
ax.set_xlabel('$\\psi_{x50}$ (MPa)')
ax.axes.get_yaxis().set_visible(False)
custom_lines = [Line2D([0], [0], linestyle='--', color='black',\
                       label='True value')]+\
               [Line2D([0], [0],\
                       color=colors[label], label=names[label])\
                for label in labels]
ax.legend(handles=custom_lines, prop={'size': 10}, frameon=False)
fig.savefig('../../../Figures/Noise.png', bbox_inches='tight')
