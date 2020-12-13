import pandas as pd
import numpy as np
from Functions import Af
import matplotlib.pyplot as plt

# read data
species_dict = {'Aru':[31, 48], 'Bpa':[56, 144], 'Pgr':[61, 122],\
                'Pst':[63, 142], 'Qru':[51, 88]}
df = pd.read_csv('../../Data/UMB_daily_average_Gil_v2.csv')

for i in range(len(df['T'])):
    T, I = df[['T', 'I']].iloc[i]
    f = lambda gs: Af(gs, T=T, I=I, Kc=460, Vcmax=63, ca=400, q=0.3, Jmax=142,\
                      z1=0.9, z2=0.9999, R=8.314)
            
    x = np.linspace(0, 1, 1001)
    y = f(x)
    y = y/max(y)
    plt.plot(x, y)
