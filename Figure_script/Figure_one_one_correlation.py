
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

### Load MCMC output
ts = pickle.load(open("../Data/45.pickle", "rb"))

# thinning
#kxmax = [item for index, item in enumerate(traces['kxmax'])
#         if index % 1 == 0]
p1 = ts['g1']
p2 = 1/ts['kxmax']

# figure
plt.scatter(p1, p2)
plt.show()
