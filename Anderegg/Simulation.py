import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import pxf3, kxf, pxminf
from scipy import optimize

# species
species = 'Bpa'
species_dict = {'Aru':['Aru_29', 31, 48], 'Bpa':['Bpa_38', 56, 144], 'Pgr':['Pgr_22', 61, 122], 'Pst':['Pst_19', 63, 142], 'Qru':['Qru_42', 51, 88]}
species_code, Vcmax, Jmax = species_dict.get(species)

# read csv
df = pd.read_csv('../Data/UMB_daily_average_Gil_2015.csv')

# extract data
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', species_code]]
df.loc[df[species_code]==0, species_code] = np.nan
df = df.dropna()
T = df['T']
I = df['I']
D = df['D']
ps = df[['ps15', 'ps30', 'ps60']].mean(1)
vn = df[species_code].reset_index()[species_code]
vn_max = vn.max()
vn = vn/vn_max

def muf(alpha, c, p50, kxmax, g1, L=1,
        ca=400, Kc=460, q=0.3, R=8.314, Jmax=Jmax, Vcmax=Vcmax, z1=0.9, z2=0.9999, a=1.6):
    sapflow_modeled = []
    for i in range(len(vn)):
        # Environmental conditions
        Ti, Ii, Di, psi, dli = T.iloc[i], I.iloc[i], D.iloc[i], ps.iloc[i], 30
        # px
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf3, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            px1 = pxf3(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            px2 = pxf3(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            if px1*px2 < 0:
                px = optimize.brentq(pxf3, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
                sapflow_modeled.append(kxf(px, kxmax, p50)*(psi-px)*30*60*dli*18/1000000/vn_max/alpha)
            else:
                if abs(px1) < abs(px2):
                    sapflow_modeled.append(1)
                else:
                    sapflow_modeled.append(0)
        else:
            print('pxmin > ps')
            sapflow_modeled.append(0)
    return sapflow_modeled

# simulation
alpha = 15
c = 7.7
g1 = 2.5
kxmax = 1300
p50 = -4.2
vn_model1 = muf(alpha, 1, p50, kxmax, g1)
vn_model2 = muf(alpha, 5, p50, kxmax, g1)
# figure
plt.plot(vn, label='observed')
plt.plot(vn_model1, label='model1')
plt.plot(vn_model2, label='model2')
plt.legend()