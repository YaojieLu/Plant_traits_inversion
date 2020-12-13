import numpy as np
import pandas as pd
from scipy import optimize
from Functions import pxf2, pxminf, kxf
import matplotlib.pyplot as plt

# species
#idx = 16
# species_dict1 = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
#                  7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
#                  17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
#                  26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
#                  32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
# codes = list(species_dict1.keys())
# code = codes[idx]
# species = species_dict1[code]
# species_code = species+'_'+str(code)

species_dict2 = {'Aru':[31, 48], 'Bpa':[56, 144], 'Pgr':[61, 122],\
                 'Pst':[63, 142], 'Qru':[51, 88]}
species_code = 'Pgr_27'
species = species_code[:3]
Vcmax, Jmax = species_dict2[species]
# read csv
df = pd.read_csv('../../Data/UMB_daily_average_Gil_v2.csv')

# extract data
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', species_code]]
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)
df = df.drop(df.index[list(range(0, 5))])
df[species_code] = df[species_code].replace({0: np.nan})
df = df.dropna()
#df = df.drop(df.index[[33, 34, 35]])
T = df['T']
I = df['I']
D = df['D']
ps = df['ps']
vn = df[species_code]

def muf(alpha_log10=-2, c=6, g1=0.5, kxmax_log10=2, p50=-1,
        ca=400, Kc=460, q=0.3, R=8.314, Vcmax=Vcmax, Jmax=Jmax, z1=0.9, z2=0.9999, a=1.6, rho=997000, L=1):
    alpha, kxmax = 10**alpha_log10, 10**kxmax_log10
    sapflow_modeled = []
    for i in range(len(vn)):
        # Environmental conditions
        Ti, Ii, Di, psi = T.iloc[i], I.iloc[i], D.iloc[i], ps.iloc[i]
        # px
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            px1 = pxf2(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            px2 = pxf2(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            if px1*px2 < 0:
                px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
                sapflow_modeled.append(kxf(px, kxmax, p50)*(psi-px)*18/1000000/1000*rho/alpha)
            else:
                sapflow_modeled.append(np.nan)
        else:
            sapflow_modeled.append(np.nan)
        print(kxf(px, kxmax, p50)*(psi-px)/(1000*a*Di))
    return sapflow_modeled

res = muf()
plt.plot(res, label='model')
plt.plot(list(vn), label='obs')
plt.legend()
#print(res)