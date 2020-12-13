import numpy as np
import pandas as pd
from scipy import optimize
from Functions import pxf2, pxminf, kxf
import matplotlib.pyplot as plt

# species
idx = 4
species_dict1 = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
codes = list(species_dict1.keys())
code = codes[idx]
species_dict2 = {'Aru':[31, 48], 'Bpa':[56, 144], 'Pgr':[61, 122], 'Pst':[63, 142], 'Qru':[51, 88]}
species = species_dict1[code]
Vcmax, Jmax = species_dict2[species]
species_code = species+'_'+str(code)
print(species_code)
# read csv
df = pd.read_csv('../../Data/UMB_daily_average_Gil_v2.csv')

# extract data
df = df[['T', 'I', 'D', 'day_len', species_code]]
df[species_code] = df[species_code].replace({0: np.nan})
df = df.dropna()
#df = df.drop(df.index[list(range(76, 79))])
T = df['T']
I = df['I']
D = df['D']
day_len = df['day_len']
vn = df[species_code]

def muf(alpha_log10=1.2, c=20, g1_log10=-0.3, kxmax_log10=1, p50=-2, ps=-1,
        Vcmax=Vcmax, Jmax=Jmax,\
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, L=1):
    alpha, g1, kxmax = 10**alpha_log10, 10**g1_log10, 10**kxmax_log10
    sapflow_modeled = []
    pxmin = pxminf(ps, p50)
    for i in range(len(T)):
        Ti, Ii, Di, dayi = T.iloc[i], I.iloc[i], D.iloc[i], day_len.iloc[i]
        # px
        pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, ps),\
                                         method='bounded',\
                                         args=(Ti, Ii, Di, ps, Kc, Vcmax, ca,\
                                               q, Jmax, z1, z2, R, g1, c,\
                                               kxmax, p50, a, L))
        px1 = pxf2(pxmin, Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R,\
                   g1, c, kxmax, p50, a, L)
        px2 = pxf2(pxmax.x, Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2,\
                   R, g1, c, kxmax, p50, a, L)
        if px1*px2 < 0:
            px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(Ti, Ii, Di, ps,\
                                                             Kc, Vcmax, ca, q,\
                                                             Jmax, z1, z2, R,\
                                                             g1, c, kxmax,\
                                                             p50, a, L))
            sapflow_modeled.append(kxf(px, kxmax, p50)*(ps-px)*30*60*18/\
                                   1000000/alpha*dayi)
        else:
            sapflow_modeled.append(np.nan)
    return sapflow_modeled

res = muf()
plt.plot(res, label='model')
plt.plot(list(vn), label='obs')
plt.legend()