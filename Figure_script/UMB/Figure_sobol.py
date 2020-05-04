
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

site = 'CZO'
df = pd.read_csv('../../Results/Sobol_day_{}.txt'.format(site), index_col=0)

# preprocessing
labels = ['$\\mathit{c}$', '$\\mathit{g_{1}}$',#'$\\alpha$', 
          '$\\mathit{k_{xmax}}$', '$\\psi_{x50}$', '$\\mathit{L}$', '$\\psi_{s}$']
df.columns = labels
df = df.stack().reset_index()
df = df.rename(columns={'level_1': 'Parameters', 0: "Sobol's total order indices"})

# figure
sns.set(style="whitegrid")
fig, ax = plt.subplots(figsize=(10, 10))
ax = sns.boxplot(ax=ax, x='Parameters', y="Sobol's total order indices", data=df)
ax.set(ylim=(0, 1.2))
ax.tick_params(labelsize=20)
ax.set_xlabel('Parameters', fontsize=20)
ax.set_ylabel("Sobol's total order indices", fontsize=20)
plt.savefig('../../Figures/Figure Sobol boxplot {} [-4, 0].png'.format(site), bbox_inches='tight')
