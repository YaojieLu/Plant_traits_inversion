
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

PLC = pd.read_csv('../Results/ensemble_PLC_UMB_Bpa.csv')
gs = pd.read_csv('../Results/ensemble_gs_UMB_Bpa.csv')
gs_max = gs['qt=0.5'].max()
# figure
species = 'Bpa'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
ax.scatter(PLC['qt=0.5'], gs['qt=0.5']/gs_max, color='k')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel('1-PLC', fontsize=30)
ax.set_ylabel('normalized gs', fontsize=30)
ax.tick_params(labelsize=20)
