import itertools
import pandas as pd
import matplotlib. pyplot as plt
import seaborn as sns

# read data
ps = ['unk', 'input']
index = ['S1', 'ST']
latex = {'c': r'$c$', 'p50': '$\\psi_{x50}$',
         'param': r'$\frac{k_{xmax}}{L \ g_1}$',
         'D': r'$D$', 'ps': '$\\psi_{s}$'}
def f(index, ps):
    x, param = [], []
    temp = pd.read_csv('../../../Results/{}_v2_ps_{}.csv'.format(index, ps))
    for c in temp.columns:
        x.extend(temp[c])
        param.extend([latex[c]]*len(temp))
    df = pd.DataFrame(list(zip(x, param)), columns=['value', 'param'])
    if index == 'S1':
        df['Index'] = pd.Series(['First-order']*len(df))
    else:
        df['Index'] = pd.Series(['Total-order']*len(df))
    df['ps'] = pd.Series([ps]*len(df))
    return df

df = None
for x, y in itertools.product(index, ps):
    temp = f(x, y)
    if df is None:
        df = temp
    else:
        df = pd.concat([df, temp])


# figures
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
sns.barplot(x='param', y='value', hue='Index', data=df[df['ps']=='unk'],\
            ax=axs[0])
axs[0].tick_params(axis='both', which='major', labelsize=30)
axs[0].set_ylim([0, 1])
axs[0].set_xlabel('')
axs[0].set_ylabel('')
axs[0].set_title('$\\psi_{s}$ as an unknown parameter', fontsize=30)
axs[0].legend(prop={'size': 30})
fig.text(0.05, 0.5, "Sobol's sensitivity index", fontsize=30,\
         va='center', rotation='vertical')
sns.barplot(x='param', y='value', hue='Index', data=df[df['ps']=='input'],\
            ax=axs[1])
axs[1].tick_params(axis='both', which='major', labelsize=30)
axs[1].set_xlabel('')
axs[1].set_ylabel('')
axs[1].set_title('$\\psi_{s}$ as an input', fontsize=30)
axs[1].get_legend().remove()
axs[1].axhline(y=0.033610)
fig.subplots_adjust(wspace=0.06)
