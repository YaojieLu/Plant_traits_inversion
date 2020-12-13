import pickle
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as grid_spec

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
colors = ['#0000ff', '#3300cc', '#660099', '#990066', '#cc0033', '#ff0000'][:3]
colors = dict(zip(labels, colors))
gs = (grid_spec.GridSpec(len(labels), 1))
fig = plt.figure(figsize=(8, 6))
i = 0
ax_objs = []
for label in labels:
    ax_objs.append(fig.add_subplot(gs[i: i+1, 0:]))

    plot = (df[df['label'] == label]['p50']\
            .plot.kde(ax=ax_objs[-1], color="#f0f0f0", lw=0.5))
    x = plot.get_children()[0]._x
    y = plot.get_children()[0]._y
    ax_objs[-1].fill_between(x, y, color=colors[label])
    
    ax_objs[-1].set_xlim(-6, 0)
    ax_objs[-1].xaxis.set_ticks_position('none')
    ax_objs[-1].axes.get_yaxis().set_visible(False)
    ax_objs[-1].text(-6.5, 0, label, fontsize=14, ha='center')
    i += 1
    
    rect = ax_objs[-1].patch
    rect.set_alpha(0)
    if i == len(labels):
        ax_objs[-1].xaxis.set_tick_params(labelsize=14)
        ax_objs[-1].set_xlabel('$\\psi_{x50}$ (MPa)', fontsize=14)
        pass
    else:
        ax_objs[-1].set_xticklabels([])
    spines = ['top', 'right', 'left', 'bottom']
    for s in spines:
        ax_objs[-1].spines[s].set_visible(False)
gs.update(hspace = -0.5)
