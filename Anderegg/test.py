
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# species information

# figure
species = 'Bpa'
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
PLC = pd.read_csv('../Results/ensemble_PLC_UMB_Bpa.csv')
gs = pd.read_csv('../Results/ensemble_gs_UMB_Bpa.csv')
ax.scatter(gs['qt=0.5'], PLC['qt=0.5'], color='k')
ax.set_xlim([0, 1])
ax.set_ylim([0, 1])
ax.set_xlabel('relative_gs', fontsize=30)
ax.set_ylabel('1-PLC', fontsize=30)

