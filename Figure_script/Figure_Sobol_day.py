
import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('../Results/Sobol_ST.txt', sep = ',', index_col = 0)
df = df.drop(['g1'], axis=1)

# figure
fig = df.plot(figsize = (8, 6))
plt.xlim([0, 200])
plt.ylim([0, 1])
fig.tick_params(labelsize = 20)
fig.set_xlabel("Days", fontsize = 20)
fig.set_ylabel("Sobol's total index", fontsize = 20)
plt.subplots_adjust(top = 0.95, bottom = 0.13)
plt.tight_layout
paras = ['$\mathit{c}$', '$\mathit{L}$', '$\mathit{\psi_{x50}}$', '$\mathit{\psi_s}$']
plt.legend(paras)
plt.savefig('../Figures/Figure Sobol_day.png')
