import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

#labels = ['baseline', 'small', 'large']
labels = ['100%', '95%', '90%', '85%', '80%', '75%']
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
plt.figure(figsize=(20, 10))
sns.set(font_scale=2)
g = sns.violinplot(x='label', y='p50', data=df, dodge=False)
plt.axhline(y=-2.5, color='red')
#g.set(xticklabels=['1.0', '0.93', '0.83'])
g.set(xticklabels=['1.0', '0.95', '0.9', '0.85', '0.8', '0.75'])
g.set(xlabel='Noise level (measured by the correlation with the original noise-free data)')
g.tick_params(bottom=False)
g.set_ylabel(ylabel='P50 (MPa)')
g.set(ylim=(-6, 0))
fig = g.get_figure()
fig.savefig('../../../Figures/noise.png', bbox_inches='tight')
# print(df.groupby('species_code')['p50'].std())