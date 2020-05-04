
import pandas as pd
import matplotlib.pyplot as plt

# read files
df_vn = pd.read_csv('../../Results/ensemble_vn_UMB_Pgr.csv')
df_vn = df_vn.set_index('date')
print(df_vn.describe())
df = pd.read_csv('../../Data/UMB_daily_average.csv')
df['date'] = pd.to_datetime((df['year']*10000+df['month']*100+df['day']).apply(str), format='%Y%m%d')
df = df[df['year']==2015]
df = df[['date', 'T', 'I', 'D', 'ps15', 'ps30']]
df = df.dropna()
df['ps'] = df[['ps15', 'ps30']].mean(1)
df = df.set_index('date')
df = df_vn.join(df)
df['err'] = (df['qt=0.5']-df['observed'])/df['observed']*100
df = df[['err', 'T', 'I', 'D', 'ps']]
print(df.err.mean())

# figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
env_labels = ['$\\mathit{T}$', '$\\mathit{I}$', '$\\mathit{D}$', '$\\psi_{s}$']
i = 0
for row in axs:
    for col in row:
        env = df.columns[i+1]
        col.scatter(df[env], df['err'])
        col.axhline(y=0)
        col.tick_params(labelsize=20)
        col.set_xlabel(env_labels[i], fontsize=30)
        if i == 0 or i == 2:
            col.set_ylabel('error (%)', fontsize=30)
        else:
            col.axes.get_yaxis().set_visible(False)
        i += 1
plt.subplots_adjust(hspace=0.25, wspace=0.05)
#plt.savefig('../../Figures/Figure_ensemble_vn_UMB.png', bbox_inches = 'tight')
