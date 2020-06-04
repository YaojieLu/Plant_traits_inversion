
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load MCMC output
species = 'Aru'
ts = pickle.load(open('../Data/UMB_trace/test.pickle'.format(species), 'rb'))

p1 = ts['b']
p2 = ts['p50']

# figure
#plt.scatter(p1, p2)
plt.hist(p2, bins=100)
plt.xlim((-3, 0))