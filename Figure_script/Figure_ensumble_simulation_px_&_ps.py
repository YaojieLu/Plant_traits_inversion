
import xlrd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read files
workbook = xlrd.open_workbook('../Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
Rf = get_data('Rf')

# import data
df_Am_px = pd.read_csv('../Results/ensemble_px_Am.csv', sep = '\t', index_col = 0)
df_Pm_px = pd.read_csv('../Results/ensemble_px_Pm.csv', sep = '\t', index_col = 0)
df_Am_ps = pd.read_csv('../Results/ensemble_ps_Am.csv', sep = '\t', index_col = 0)
df_Pm_ps = pd.read_csv('../Results/ensemble_ps_Pm.csv', sep = '\t', index_col = 0)

# figure
labels = ['Pacific Madrone', 'Douglas fir']
fig = plt.figure(figsize = (20, 12))

# ps
# Am
ax1 = fig.add_subplot(2, 2, 1)
plt.plot(df_Am_ps.loc[:, 'qt = 0.5'], color = 'k')
plt.fill_between(df_Am_ps.index, df_Am_ps.loc[:, 'qt = 0.05'], df_Am_ps.loc[:, 'qt = 0.95'],
                 color = 'b', alpha = 0.2)
plt.xlim([0, 200])
plt.ylim([-8, 0])
plt.ylabel('$\psi_{s}$ (MPa)', fontsize = 20)
plt.title(labels[0], fontsize = 20)
ax1.tick_params(labelsize = 20)
ax1.tick_params(axis = 'x', which = 'major', pad = 5)
ax1.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
# rainfall
ax12 = ax1.twinx()
ax12.plot(Rf, color = 'r')
plt.ylim([0, 150])
ax12.tick_params(axis = 'y', which = 'both', right = False, labelright = False)
# Pm
ax2 = fig.add_subplot(2, 2, 2)
plt.plot(df_Pm_ps.loc[:, 'qt = 0.5'], color = 'k')
plt.fill_between(df_Pm_ps.index, df_Pm_ps.loc[:, 'qt = 0.05'], df_Pm_ps.loc[:, 'qt = 0.95'],
                 color = 'b', alpha = 0.2)
plt.xlim([0, 200])
plt.ylim([-8, 0])
plt.title(labels[1], fontsize = 20)
ax2.tick_params(labelsize = 20)
ax2.tick_params(axis = 'x', which = 'both', bottom = False, labelbottom = False)
ax2.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
# rainfall
ax22 = ax2.twinx()
ax22.plot(Rf, color = 'r')
plt.ylim([0, 150])
ax22.set_ylabel('Rainfall (mm)', color = 'r', fontsize = 20)
ax22.set_yticks(np.linspace(0, 150, 4))
ax22.tick_params(axis = 'y', labelsize = 20, labelcolor = 'r')

# px
# Am
ax3 = fig.add_subplot(2, 2, 3)
plt.plot(df_Am_px.loc[:, 'qt = 0.5'], color = 'k')
plt.fill_between(df_Am_px.index, df_Am_px.loc[:, 'qt = 0.05'], df_Am_px.loc[:, 'qt = 0.95'],
                 color = 'b', alpha = 0.2)
plt.xlim([0, 200])
plt.ylim([-8, 0])
plt.xlabel("Days", fontsize = 20)
plt.ylabel('$\psi_{x}$ (MPa)', fontsize = 20)
ax3.tick_params(labelsize = 20)
ax3.tick_params(axis = 'x', which = 'major', pad = 5)
# rainfall
ax32 = ax3.twinx()
ax32.plot(Rf, color = 'r')
plt.ylim([0, 150])
ax32.tick_params(axis = 'y', which = 'both', right = False, labelright = False)
# Pm
ax4 = fig.add_subplot(2, 2, 4)
plt.plot(df_Pm_px.loc[:, 'qt = 0.5'], color = 'k')
plt.fill_between(df_Pm_px.index, df_Pm_px.loc[:, 'qt = 0.05'], df_Pm_px.loc[:, 'qt = 0.95'],
                 color = 'b', alpha = 0.2)
plt.xlim([0, 200])
plt.ylim([-8, 0])
plt.xlabel("Days", fontsize = 20)
ax4.tick_params(labelsize = 20)
ax4.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
# rainfall
ax42 = ax4.twinx()
ax42.plot(Rf, color = 'r')
plt.ylim([0, 150])
ax42.set_ylabel('Rainfall (mm)', color = 'r', fontsize = 20)
ax42.set_yticks(np.linspace(0, 150, 4))
ax42.tick_params(axis = 'y', labelsize = 20, labelcolor = 'r')

# adjustment
plt.tight_layout
plt.subplots_adjust(wspace = 0.1, hspace = 0.1)
 
# save
plt.savefig('../Figures/Figure ensemble simlation px & ps.png', bbox_inches = 'tight')
