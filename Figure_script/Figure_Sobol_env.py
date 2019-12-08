
import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('../Results/Sobol2_ST.txt', sep = ',', index_col = 0)
df_T = df[df['T'] == 30]

# labels
paras = ['c', 'g1', 'L', 'p50', 'ps']
latex = ['$\\mathit{c}$', '$\\mathit{g_1}$', '$\\mathit{L}$',
         '$\\psi_{x50}$', '$\\psi_{s}$']
labels = dict(zip(paras, latex))

# figure
fig = plt.figure(figsize = (40, 12))
for i in range(len(paras)):
    ax = fig.add_subplot(2, len(paras), i+1)
    plt.scatter(df_T['I'], df_T[paras[i]], c = df_T['D'])
    plt.ylim([0, 1])
    plt.xlabel('$\\mathit{I}$', fontsize = 20)
    plt.title(labels[paras[i]], fontsize = 20)
    if i == 0:
        plt.ylabel("Sobol's total index", fontsize = 20)
    else:
        ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout

for i in range(len(paras)):
    ax = fig.add_subplot(2, len(paras), i+1+len(paras))
    plt.scatter(df_T['D'], df_T[paras[i]], c = df_T['I'])
    plt.ylim([0, 1])
    plt.xlabel('$\\mathit{D}$', fontsize = 20)
    if i == 0:
        plt.ylabel("Sobol's total order indices", fontsize = 20)
    else:
        ax.axes.get_yaxis().set_visible(False)
    plt.tight_layout
plt.subplots_adjust(wspace = 0.1)
plt.savefig('../Figures/Figure Sobol_env.png', bbox_inches = 'tight')
