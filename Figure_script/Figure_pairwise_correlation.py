
import numpy as np
import pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

# load MCMC output and thin
ts = pickle.load(open("../Data/45.pickle", "rb"))
params = ['c', 'p50']
c = np.asarray([item for index, item in enumerate(ts['c']) if index % 1 == 0])
p50 = np.asarray([item for index, item in enumerate(ts['p50']) if index % 1 == 0])
c, p50 = c.reshape(-1, 1), p50.reshape(-1, 1)
reg = LinearRegression().fit(c, p50)
print(reg.coef_, reg.intercept_)

# figure
plt.scatter(c, p50)