
import pandas as pd

# no_root frequency
df2 = pd.read_csv('bad_days.txt', header=None, names=['1', '2'], sep=',')
df2 = df2[~df2['1'].str.contains('%|Plotting')]
total_iter = 1000000

# frequency
df_num = df2[df2['2'].isna()][['1']]
df_num = df_num['1'].value_counts()
df_num = df_num.to_frame().reset_index()
df_num.columns = ['num', 'freq']
df_num = df_num.sort_values(by=['num'])
df_num['freq'] = df_num['freq']/total_iter
df_num['freq'] = df_num['freq'].fillna(0)
print(df_num.sort_values(by=['freq'], ascending=False).head(10))

# parameters
df3 = df2[df2['2'].notna()]
df_par = df3[['1', '2']].drop_duplicates()
df_par.columns = ['c', 'p50']
df_par = df_par.astype('float64')
ax = df_par.plot.scatter('c', 'p50')
ax.set_title('{:0.0f} out of {} iterations'.format(len(df_par), total_iter), fontsize=20)