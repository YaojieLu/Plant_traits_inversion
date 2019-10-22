
import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read files
df = pd.read_csv('../Results/ensemble_Am.csv', sep = '\t', index_col = 0)
workbook = xlrd.open_workbook('../Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
vn = get_data('Am_RN_N')

# figure
fig = plt.figure(figsize = (10, 6))
ax1 = fig.add_subplot(1, 1, 1)
plt.plot(df.loc[:, 'qt = 0.5'], color = 'k')
plt.plot(vn, color = 'lightcoral')
plt.fill_between(df.index, df.loc[:, 'qt = 0.05'], df.loc[:, 'qt = 0.95'],
                 color = 'b', alpha = 0.2)
plt.xlim([0, 200])
plt.ylim([0, 1])
plt.xlabel("Days", fontsize = 20)
plt.ylabel("Sap velocity", fontsize = 20)
ax1.tick_params(labelsize = 20)
plt.tight_layout

ax1.legend(['ensemble simulation', 'input'], loc = 'upper left', fontsize = 20, framealpha = 0)
plt.savefig('../Figures/Figure ensemble simulation vn Am.png', bbox_inches = 'tight')
