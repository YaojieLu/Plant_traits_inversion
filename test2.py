import pandas as pd
from Functions import pxf, pxminf
import matplotlib.pyplot as plt
import numpy as np
from scipy import optimize

# species
species = 'Am'
species_dict = {'Am':'Am_RN_N', 'Nd':'Nd_RN_S', 'Pm':'Pm_RN_S', 'Qc':'Qc_RN_S', 'Qg':'Qg_SS_S'}
species_code = species_dict.get(species)

# read csv
df = pd.read_csv('Data/Dataset.csv')

# extract data
df = df[['T', 'I', 'D', species_code]].iloc[10]
T = df['T']
I = df['I']
D = df['D']
vn = df[species_code]
vn_max = vn.max()
vn = vn/vn_max

kxmax = 7
p50 = -5
ps = -1
g1=10
f = lambda px:pxf(px, T, I, D, ps=ps, Kc=460, Vcmax=30, ca=400, q=0.3, Jmax=80, z1=0.9, z2=0.9999, R=8.314, g1=g1, c=30, kxmax=kxmax, p50=p50, a=1.6, L=1)
x_min = pxminf(ps, p50)
x_max = optimize.minimize_scalar(f, bounds=(x_min, ps), method='bounded').x
x = np.linspace(x_min, x_max, 10)
y = f(x)

plt.plot(y)
plt.axhline(y=0, color='red')