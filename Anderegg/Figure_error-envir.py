import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

species_code = 'Bpa_38'
df = pd.read_csv('../Data/UMB_daily_average.csv')
#df = df[df['year']==2015]
df['date'] = pd.to_datetime((df['year']*10000+df['month']*100+df['day'])\
                            .apply(str), format='%Y%m%d')
df = df[['date', 'T', 'I', 'D', 'ps15', 'ps30', 'ps60', species_code]]
df.loc[df[species_code]==0, species_code] = np.nan
df = df.dropna()
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)
df['modelled'] = list(pd.read_csv('../Results/ensemble_vn_UMB_Bpa.csv')['qt=0.5'])
df['err'] = abs(df[species_code]-df['modelled'])

# figure
fig, axs = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
envs = ['T', 'I', 'D', 'ps']
i = 0
for row in axs:
    for col in row:
        env = envs[i]
        col.scatter(df[env], df['err'], color='k')
        col.tick_params(labelsize=20)
        col.set_xlabel(env, fontsize=30)
        if i == 0 or i == 2:
            col.set_ylabel('abs err', fontsize=30)
        else:
            col.axes.get_yaxis().set_visible(False)
        i += 1
plt.subplots_adjust(hspace=0.3, wspace=0.1)
fig.suptitle('Model-data deviation', fontsize=30)
