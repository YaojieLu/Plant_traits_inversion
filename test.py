
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import data
df = pd.read_csv('Results/Sobol_env.txt', sep = ',', index_col = 0)
df = df[df['T'] == 30]
df['D'] = round(df['D'], 4)

# labels
paras = ['c']
latex = ['$\\mathit{c}$']
labels = dict(zip(paras, latex))

# figure
sns.set(font_scale = 1.3)
fig = plt.figure(figsize = (16, 16))
df_para = df.pivot(index = 'I', columns = 'D', values = 'c')
sns.heatmap(df_para, cmap = 'viridis', xticklabels = 3, yticklabels = 3)
    
