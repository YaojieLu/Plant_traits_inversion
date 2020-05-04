
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Simulation_model import UMBf, UMBf2

# read csv
df = pd.read_csv('../Data/UMB_daily_average.csv')
T = df['T']
I = df['I']
D = df['D']
ps = df['ps']
vn_obs = df['Aru_29']

# simulation
alpha = 0.0056
c = 2.302207
p50 = -2.974111
kxmax = 1.852892
g1 = 12.756095
L = 0.671456
vn1 = UMBf([alpha, c, p50, kxmax, g1, L], T, I, D, ps)
vn2 = UMBf2([alpha, c, p50, kxmax, 1, L], T, I, D, ps)

# figure
plt.plot(vn_obs, label='Aru_29')
plt.plot(vn1, label='Original')
plt.plot(vn2, label='Medlyn')
plt.legend()