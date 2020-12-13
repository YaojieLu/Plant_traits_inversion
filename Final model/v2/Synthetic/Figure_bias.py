import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

labels = ['ps_1', 'ps_2', 'ps_3']
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
plt.figure(figsize=(9, 9))
sns.set(font_scale=2)
sns.set_style('white')
g = sns.violinplot(x='label', y='p50', data=df)
plt.axhline(y=-2.5, color='red')
g.set(xticklabels=['$\\psi_{s_{measured}}$ > $\\psi_{s_{true}}$',\
                   '$\\psi_{s_{measured}}$ = $\\psi_{s_{true}}$',\
                   '$\\psi_{s_{measured}}$ < $\\psi_{s_{true}}$'])
g.set(xlabel='')
g.tick_params(bottom=False)
g.set_ylabel(ylabel='P50 (MPa)')
g.set(ylim=(-5, 0))
fig = g.get_figure()
fig.savefig('../../../Figures/bias_ps.png', bbox_inches='tight')
