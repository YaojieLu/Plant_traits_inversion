import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2
import matplotlib.pyplot as plt

# species infomation dict
species = 'Aru'
species_dict = {'Aru':['Aru_29', 31, 48], 'Bpa':['Bpa_38', 56, 144],
                'Pgr':['Pgr_27', 61, 122], 'Pst':['Pst_2', 63, 142],
                'Qru':['Qru_10', 51, 88]}
'pgr: 3, 4, 22, 34'
'aru: 9, 21, 24, 26, 28'
# input
species_code, Vcmax, Jmax = 'Pgr_34', 61, 122

# read csv
df = pd.read_csv('../Data/UMB_daily_average.csv')
df = df[df['year']==2015]

# extract data
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', species_code]]
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)
df.loc[df[species_code]==0, species_code] = np.nan
df = df.dropna()
df = df.drop(df.index[list(range(76, 79))])
T = df['T']
I = df['I']
D = df['D']
ps = df['ps']
vn = df[species_code]

def muf(alpha_log10=1.2, c=1, g1=20, kxmax_log10=3, p50=-1,
        T=T, I=I, D=D, ps=ps, Jmax=Jmax, Vcmax=Vcmax,\
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, L=1):
    alpha, kxmax = 10**alpha_log10, 10**kxmax_log10
    sapflow_modeled = []
    for i in range(len(T)):
        Ti, Ii, Di, psi = T.iloc[i], I.iloc[i], D.iloc[i], ps.iloc[i]
        # px
        pxmin = pxminf(psi, p50)
        pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, psi),\
                                         method='bounded',\
                                         args=(Ti, Ii, Di, psi, Kc, Vcmax, ca,\
                                               q, Jmax, z1, z2, R, g1, c,\
                                               kxmax, p50, a, L))
        px1 = pxf2(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R,\
                   g1, c, kxmax, p50, a, L)
        px2 = pxf2(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2,\
                   R, g1, c, kxmax, p50, a, L)
        if px1*px2 < 0:
            px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(Ti, Ii, Di, psi,\
                                                             Kc, Vcmax, ca, q,\
                                                             Jmax, z1, z2, R,\
                                                             g1, c, kxmax,\
                                                             p50, a, L))
            sapflow_modeled.append(kxf(px, kxmax, p50)*(psi-px)*30*60*18/\
                                   1000000/alpha)
        else:
            sapflow_modeled.append(np.nan)
    return sapflow_modeled

res = muf()
plt.plot(res)
plt.plot(list(vn))