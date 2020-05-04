
import pandas as pd
import matplotlib.pyplot as plt

# no_root frequency
df = pd.read_csv('../Data/UMB_daily_average.csv')
df.rename(columns={df.columns[0]: 'num'}, inplace=True)
df2 = pd.read_csv('../Data/bad_days.txt', header=None, names=['num'])
df2 = df2[pd.to_numeric(df2['num'], errors='coerce').notnull()]
df3 = df2['num'].value_counts()
df3 = df3.to_frame().reset_index()
df3.columns = ['num', 'freq']
df3 = df3.sort_values(by=['num'])
df3['num'] = [df['num'][int(i)] for i in df3['num']]
df3 = df3.astype('int64')
df4 = pd.merge(df, df3, how='outer', on=['num'])
df4['freq'] = df4['freq']/1000000
df4['freq'] = df4['freq'].fillna(0)
df4 = df4.drop(['num'], axis=1)

# filtered dataset
freq_threshold = 0.1
df5 = df4[df4['freq']<freq_threshold]
print(len(df5)/len(df4))
print(df4.index[df4['freq']>freq_threshold])
df5.to_csv('../Data/UMB_daily_average_filtered.csv', sep = ',')

# figure
envs = ['T', 'I', 'D', 'ps']
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
i = 0
for row in axs:
    for col in row:
        x, y = df4['freq'], df4[envs[i]]
        col.scatter(x, y)
        if i < 2:
            col.axes.get_xaxis().set_visible(False)
        else:
            col.set_xlabel('Frequency', fontsize=30)
        col.set_ylabel(envs[i], fontsize=30)
        col.set_title(r'$R^{}$ = {:0.2f}'.format(2, x.corr(y)), fontsize=30)
        i += 1
plt.subplots_adjust(wspace=0.15, hspace=0.2)
