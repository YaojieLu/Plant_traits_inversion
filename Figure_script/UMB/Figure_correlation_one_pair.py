
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load MCMC output
species = 'Aru'
ts = pickle.load(open('../../Data/UMB_trace/{}.pickle'.format(species), 'rb'))

p1 = ts['kxmax']
p2 = ts['alpha']

# figure
plt.scatter(p1, p2)
plt.xlabel('$\\mathit{k_{xmax}}$', fontsize=30)
plt.ylabel('$\\alpha$', fontsize=30)
plt.savefig('../../Figures/Figure UMB alpha vs kxmax.png', bbox_inches='tight')
