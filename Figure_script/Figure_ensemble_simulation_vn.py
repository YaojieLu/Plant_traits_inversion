
import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read files
workbook = xlrd.open_workbook('../Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
dictsp = {'Am':'Am_RN_N', 'Nd':'Nd_RN_S', 'Pm':'Pm_RN_S',
          'Qc':'Qc_RN_S', 'Qg':'Qg_SS_S', 'Syn':'Synthetic'}
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
vn_Am = get_data(dictsp.get('Am'))
vn_Pm = get_data(dictsp.get('Pm'))

# import data
df_Am = pd.read_csv('../Results/ensemble_vn_Am.csv', sep = '\t', index_col = 0)
df_Pm = pd.read_csv('../Results/ensemble_vn_Pm.csv', sep = '\t', index_col = 0)

# figure
labels = ['Pacific Madrone', 'Douglas fir']
fig = plt.figure(figsize = (20, 6))
# Am
ax1 = fig.add_subplot(1, 2, 1)
plt.plot(df_Am.loc[:, 'qt = 0.5'], color = 'k')
plt.plot(vn_Am, color = 'lightcoral')
plt.fill_between(df_Am.index, df_Am.loc[:, 'qt = 0.05'], df_Am.loc[:, 'qt = 0.95'],
                 color = 'b', alpha = 0.2)
plt.xlim([0, 200])
plt.ylim([0, 1.4])
plt.xlabel("Days", fontsize = 20)
plt.ylabel("Sap velocity", fontsize = 20)
plt.title(labels[0], fontsize = 20)
ax1.tick_params(labelsize = 20)
plt.tight_layout
# Pm
ax2 = fig.add_subplot(1, 2, 2)
plt.plot(df_Pm.loc[:, 'qt = 0.5'], color = 'k')
plt.plot(vn_Pm, color = 'lightcoral')
plt.fill_between(df_Pm.index, df_Pm.loc[:, 'qt = 0.05'], df_Pm.loc[:, 'qt = 0.95'],
                 color = 'b', alpha = 0.2)
plt.xlim([0, 200])
plt.ylim([0, 1.4])
plt.xlabel("Days", fontsize = 20)
plt.ylabel("Sap velocity", fontsize = 20)
plt.title(labels[1], fontsize = 20)
ax2.tick_params(labelsize = 20)
plt.tight_layout

ax1.legend(['modelled', 'observed'], loc = 'upper left', fontsize = 20, framealpha = 0)
plt.savefig('../Figures/Figure ensemble simlation vn.png', bbox_inches = 'tight')
