import pickle
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# load traces
ts = pickle.load(open("../Data/45.pickle", "rb"))
params = ['g1', 'kxmax', 'L', 'alpha', 'c', 'p50']
df = {}
for key in params:
    df[key] = [item for index, item in enumerate(ts[key]) if index % 15 == 0]
df = pd.DataFrame.from_dict(data = df, orient = 'columns')
df['L*g1'] = df['L']*df['g1']
df['kxmax/alpha'] = df['kxmax']/df['alpha']
df = df.drop(['L', 'g1', 'alpha', 'kxmax'], axis=1)
params = ['L*g1', 'kxmax/alpha', 'c', 'p50']
true_values = [100, 350, 16, -4.5]

# figure
ranges = [[5, 500], [5, 1000], [2, 20], [-10, -0.1]]
fig, axs = plt.subplots(nrows=1, ncols=4, figsize=(40, 10))
for idx, col in enumerate(axs):
    param = params[idx]
    df_temp = pd.DataFrame({param: df[param]}).iloc[:, 0]
    col.hist(df_temp, range=[ranges[idx][0], ranges[idx][1]], bins=100)
    mean, std = df_temp.mean(), df_temp.std()
    cv = abs(round(std/mean, 2))
    col.set_title('RSD = {}'.format(cv), fontsize=30)
    col.axvline(x=true_values[idx], c='black',
                label='True value', linestyle='dashed')
    col.axes.get_yaxis().set_visible(False)
    col.tick_params(labelsize=30)
    col.set_xlabel(params[idx], fontsize=30)
    if idx == 0:
        col.legend([Line2D([0], [0], linestyle='dashed', color='black')],
                    ['True value'], loc='upper right', fontsize=30, framealpha=0)

plt.subplots_adjust(hspace=0.3, wspace=0.1)
