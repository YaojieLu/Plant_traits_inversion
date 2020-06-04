import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import pxf2, kxf, pxminf
from scipy import optimize

# species
species = 'Aru'
species_dict = {'Aru':['Aru_29', 31, 48], 'Bpa':['Bpa_38', 56, 144], 'Pgr':['Pgr_22', 61, 122], 'Pst':['Pst_19', 63, 142], 'Qru':['Qru_42', 51, 88]}
species_code, Vcmax, Jmax = species_dict.get(species)

# read csv
df = pd.read_csv('../Data/UMB_daily_average.csv')
df = df[df['year']==2015.0]

# extract data
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', species_code]]
df.loc[df[species_code]==0, species_code] = np.nan
df = df.dropna()
T = df['T']
I = df['I']
D = df['D']
ps = df[['ps15', 'ps30', 'ps60']].mean(1)
vn = df[species_code].reset_index()[species_code]

def UMBf(X, T, I, D, ps,
         ca=400, Kc=460, q=0.3, R=8.314, Jmax=48, Vcmax=31, z1=0.9, z2=0.9999,
         a=1.6, l=1.8*10**(-5), u=48240):   
    alpha, c, p50, kxmax, g1, L = X
    sapflow_modeled = []
    for i in range(len(T)):
        Ti, Ii, Di, psi = T.iloc[i], I.iloc[i], D.iloc[i], ps.iloc[i]
        # px
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            px1 = pxf2(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            px2 = pxf2(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            if px1*px2 < 0:
                px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            else:
                if abs(px1) < abs(px2):
                    px = pxmin
                else:
                    px = pxmax.x
            # Soil water balance
            sapflow_modeled.append(l*u*kxf(px, kxmax, p50)*(psi-px)/1000/alpha)        
        else:
            print('pxmin > ps')
            sapflow_modeled.append(0)
    return sapflow_modeled

# simulation
alpha = 0.00890493730691395
c = 23.0873290851329
p50 = -0.541318043759336
kxmax = 8.077336183819579
g1 = 0.7356126247253332
L = 0.8504314546720629
vn_model = UMBf([alpha, c, p50, kxmax, g1, L], T, I, D, ps)

# figure
plt.plot(vn, label='Aru_29')
plt.plot(vn_model, label='model')
plt.legend()