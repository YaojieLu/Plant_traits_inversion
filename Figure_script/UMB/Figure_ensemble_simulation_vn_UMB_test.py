
import pandas as pd
import matplotlib.pyplot as plt

# script inputs
run, species = 16, 'Pgr'

# figure
fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 20))
df = pd.read_csv('../../Results/ensemble_vn_UMB_{}_{}.csv'.format(run, species))
df = df[df['qt=0.95'] <= 1]
ax.plot(df['qt=0.5'], color='k')
ax.plot(df['vn'], color='lightcoral')
ax.fill_between(df.index, df['qt=0.05'], df['qt=0.95'],
                 color='b', alpha=0.2)
#col.set_xlim([0, 103])
#col.set_ylim([0, 1])
ax.set_title(species, fontsize=30)
ax.set_title(r'{}; $R^{}$ = {:0.2f}'.format(species, 2, df['qt=0.5'].corr(df['vn'])), fontsize=30)
ax.tick_params(labelsize=20)
ax.set_xlabel('Day', fontsize=30)
ax.set_ylabel('Sap velocity', fontsize=30)
ax.legend(['model', 'data'], loc='upper left', fontsize=20, framealpha=0)
