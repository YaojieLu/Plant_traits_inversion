
import xlrd
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import data
workbook = xlrd.open_workbook('../Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
D = np.asarray(sheet.col_values(np.where(keys=='D')[0][0])[1:])
df = pd.read_csv('../Results/Sobol_day_7.txt', sep = ',', index_col = 0)
df['D'] = D
df = df[df['D']>0.001]
#df = df[['c', 'p50', 'ps']]
df.drop(columns='D', inplace=True)
print(df.mean())
# preprocessing
labels = ['$\\alpha$', '$\\mathit{c}$', '$\\mathit{g_{1}}$', '$\\mathit{k_{xmax}}$', '$\\psi_{x50}$', '$\\mathit{L}$', '$\\psi_{s}$']
#labels = ['$\\mathit{c}$', '$\\psi_{x50}$', '$\\psi_{s}$']
df.columns = labels
df = df.reset_index()
df = df.set_index('Days').stack().reset_index()
df = df.rename(columns={'level_1': 'Parameters', 0: "Sobol's total order indices"})

# figure
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(ax=ax, x='Parameters', y="Sobol's total order indices", data=df)
ax.set(ylim=(0, 1.2))
ax.tick_params(labelsize=20)
ax.set_xlabel('Parameters', fontsize=20)
ax.set_ylabel("Sobol's total order indices", fontsize=20)
plt.savefig('../Figures/Figure Sobol boxplot 7.png', bbox_inches = 'tight')
