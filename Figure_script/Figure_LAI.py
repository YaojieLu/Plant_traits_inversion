
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
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:61])
vn_Am = get_data(dictsp.get('Am'))

# import data
df_const_LAI = pd.read_csv('../Results/ensemble_vn_Am_constant_LAI.csv',
                           sep = '\t', index_col = 0)
df_sin_LAI = pd.read_csv('../Results/ensemble_vn_Am.csv',
                           sep = '\t', index_col = 0)

# figure
fig = plt.figure(figsize = (8, 6))
# Am
ax = fig.add_subplot(1, 1, 1)
plt.scatter(vn_Am, df_const_LAI.loc[:, 'qt = 0.5'], color = 'r')
plt.scatter(vn_Am, df_sin_LAI.loc[:59, 'qt = 0.5'], color = 'b')
plt.plot([0, 0.5], [0, 0.5], 'k--', lw = 1)

plt.xlim([0, 0.5])
plt.ylim([0, 0.5])
plt.xlabel("Observed sap velocity", fontsize = 20)
plt.ylabel("Modelled sap velocity", fontsize = 20)
ax.tick_params(labelsize = 20)
plt.tight_layout

ax.legend(['1:1', 'Constant LAI', 'Variable LAI'], loc = 'upper left', fontsize = 20, framealpha = 0)
plt.savefig('../Figures/Figure constant LAI.png', bbox_inches = 'tight')
