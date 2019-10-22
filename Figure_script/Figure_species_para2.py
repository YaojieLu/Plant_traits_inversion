import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats

# load traces
para = 'p50'
Am = pickle.load(open("../Data/Am.pickle", "rb"))[para]# traces baseline
Pm = pickle.load(open("../Data/Pm.pickle", "rb"))[para]# traces given L
ts_df = pd.DataFrame({'Am': Am, 'Pm': Pm})

# kernel density estimation
x = np.linspace(min(Am.min(), Pm.min()), max(Am.max(), Pm.max()), 1000)
kde_Am = stats.gaussian_kde(ts_df.iloc[:, 0])(x)
kde_Pm = stats.gaussian_kde(ts_df.iloc[:, 1])(x)

# figure
labels = ['Pacific Madrone', 'Douglas fir']
fig = plt.figure(figsize = (8, 6))
plt.plot(x, kde_Am, marker = '', color = 'k')
plt.plot(x, kde_Pm, marker = '', color = 'b')
#plt.hist(x, ts_df.loc[:, 'Am'])
plt.tick_params(axis = 'x')
plt.tick_params(axis = 'y', which = 'both', left = False, labelleft = False)
plt.legend(labels, loc = 'best')
plt.tight_layout
