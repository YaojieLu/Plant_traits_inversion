
import pandas as pd
import matplotlib.pyplot as plt

# import data
df = pd.read_csv('../Results/Sobol_env.txt', sep = ',', index_col = 0)
df['D'] = round(df['D'], 4)

# figure
labels = ['$\\mathit{c}$', '$\\mathit{k}_{xmax}$', '$\\psi_{x50}$', 
         '$\\mathit{L}$', '$\\psi_{s}$']
D = [0.0035, 0.03, 0.0035, 0.03]
T = [5, 5, 30, 30]
fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(20, 20))
scenario = 0
for i, row in enumerate(ax):
    for j, col in enumerate(row):
        T_temp, D_temp = T[scenario], D[scenario]
        df_temp = df[(df['T']==T_temp) & (df['D']==D_temp)]
        col.plot(df_temp['I'], df_temp[['c', 'kxmax', 'p50', 'L', 'ps']])
        col.set_title('$D$ = {} kPa; $T$ = {} $^\circ$C'.
                      format(round(D_temp*101.325, 2), T_temp), fontsize=30)
        col.set_xlim([0, 500])
        col.set_ylim([0, 1.1])
        col.tick_params(labelsize=30)
        if i == 1:
            col.set_xlabel('$\\mathit{I}$', fontsize=30)
        else:
            col.axes.get_xaxis().set_visible(False)
        if j == 0:
            col.set_ylabel("Sobol's total order indices", fontsize=30)
        else:
            col.axes.get_yaxis().set_visible(False)
        scenario += 1
# legend
fig.legend(labels=labels, loc='center right', borderaxespad=0.1, fontsize=30)
plt.subplots_adjust(wspace=0.1, hspace=0.1, right=0.85)
plt.savefig('../Figures/Figure Sobol_env_line.png', bbox_inches = 'tight')
