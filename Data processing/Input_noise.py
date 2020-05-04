
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# read csv file
df = pd.read_csv('../Data/raw_data/UMB_rawdata.csv')
df = df.replace(r'^\s*$', np.nan, regex=True)

# functions
def f1(env):
    x = df[env]
    xre = x.values.reshape(-1, 48)
    mean = np.apply_along_axis(np.nanmean, 0, xre)
    std = np.apply_along_axis(np.nanstd, 0, xre)
    lower, upper = mean-std, mean+std
    res = [mean, lower, upper]
    return res
# coefficient of variance
def cov_f(env):
    x = df[env]
    xre = x.values.reshape(-1, 48)
    grand_mean = np.nanmean(xre)
    res = np.apply_along_axis(np.nanstd, 0, xre)/grand_mean
    return res

# calculation
envs = ['T', 'I', 'D']
stats = ['mean', 'lower', 'upper']

df_cov = pd.DataFrame([cov_f(env) for env in envs], index=envs).T
df_T = pd.DataFrame(f1('T'), index=stats).T
df_I = pd.DataFrame(f1('I'), index=stats).T
df_D = pd.DataFrame(f1('D'), index=stats).T
df_list = [df_T, df_I, df_D]

# figure
fig, axs = plt.subplots(1, 4, figsize=(40, 5))
for i in range(3):
    df = df_list[i]
    ax = axs[i]
    df.plot(ax=ax, color=['black', 'r', 'r'], title=envs[i], legend=False)
df_cov.plot(ax=axs[3], title='Coefficient of variance')