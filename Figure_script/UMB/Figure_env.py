
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# read files
df = pd.read_csv('../../Data/UMB_daily_average.csv')
df['date'] = pd.to_datetime((df['year']*10000+df['month']*100+df['day']).apply(str), format='%Y%m%d')
df = df[df['year']==2015]
df = df[['date', 'T', 'I', 'D', 'ps15', 'ps30']]
df['D'] = df['D']*100
df['ps'] = df[['ps15', 'ps30']].mean(1)

# figure
fig, axs = plt.subplots(nrows=4, ncols=1, figsize=(40, 80))
env = ['T', 'I', 'D', 'ps']
env_labels = ['$\\mathit{T}$'+'\n'+'$(^oC)$', '$\\mathit{I}$'+'\n'+'$(W/m^2)$',
              '$\\mathit{D}$'+'\n'+'(kPa)', '$\\psi_{s}$'+'\n'+'(MPa)']
ylim = [[-4, 40], [-100, 1000], [-0.4, 4], [-2, 0.2]]
i = 0
for row in axs:
    row.plot(df['date'], df[env[i]])
    row.tick_params(labelsize=20)
    row.set_ylim(ylim[i])
    row.set_ylabel(env_labels[i], fontsize=30, rotation=0)
    row.yaxis.set_label_coords(-0.09, 0.25)
    if i == 3:
        row.set_xlabel('Date', fontsize=30)
        row.xaxis.set_major_locator(ticker.MultipleLocator(15))
        row.xaxis.set_tick_params(rotation=20)
    else:
        row.axes.get_xaxis().set_visible(False)
    i += 1
plt.subplots_adjust(hspace=0.1, wspace=0)
plt.savefig('../../Figures/Figure_env_UMB.png', bbox_inches='tight')
