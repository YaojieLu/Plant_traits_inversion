
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import data
df = pd.read_csv('../Results/Sobol_env.txt', sep = ',', index_col = 0)
df = df[df['T'] == 30]
df['D'] = round(df['D'], 4)

# labels
paras = ['c', 'L', 'p50', 'ps']
latex = ['$\\mathit{c}$', '$\\mathit{L}$',
         '$\\psi_{x50}$', '$\\psi_{s}$']
labels = dict(zip(paras, latex))

# figure
sns.set(font_scale = 1.3)
fig = plt.figure(figsize = (16, 16))
for i in range(len(paras)):
    ax = fig.add_subplot(2, len(paras)/2, i+1)
    df_para = df.pivot(index = 'I', columns = 'D', values = paras[i])
    sns.heatmap(df_para, cmap = 'viridis', xticklabels = 3, yticklabels = 3)
    #plt.xlim
    #plt.ylim([0, 1])
    
    if i > 1:
        plt.xlabel('$\\mathit{D}$', fontsize = 20)
    else:
        ax.axes.get_xaxis().set_visible(False)
    if i == 0 or i == 2:
        plt.ylabel('$\\mathit{I}$', fontsize = 20)
    else:
        ax.axes.get_yaxis().set_visible(False)
    plt.title(labels[paras[i]], fontsize = 20)        
    plt.tight_layout
plt.subplots_adjust(wspace = 0, hspace = 0.15)
plt.savefig('../Figures/Figure Sobol_env_heatmap.png', bbox_inches = 'tight')
