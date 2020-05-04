
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read files
df = pd.read_csv('../Data/UMB_daily_average.csv')

# figure
envs = ['T', 'I', 'D', 'ps']
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
i = 0
for row in axs:
    for col in row:
        x, y = df[envs[i]], df['Aru_29']
        col.scatter(x, y)
        col.set_xlabel(envs[i], fontsize=30)
        col.set_ylabel('vn', fontsize=30)
        col.set_title(r'$R^{}$ = {:0.2f}'.format(2, x.corr(y)), fontsize=30)
        i += 1
plt.subplots_adjust(wspace=0.15, hspace=0.4)
