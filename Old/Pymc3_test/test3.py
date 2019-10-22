import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# Draw 1 million samples from an exponential distribution with lambda = 1/10
x = np.random.exponential(10, 1000000)

# Plot a histogram
plt.hist(x, bins = 100)
plt.savefig('graph.png')
#plt.show()

# Calulate mean and variance
print(np.mean(x), np.var(x))