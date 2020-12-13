import pandas as pd
from Functions import pxf2, pxminf, kxf
from scipy import optimize
import matplotlib.pyplot as plt

# species
species = 'Am'
species_dict = {'Am':'Am_RN_N', 'Nd':'Nd_RN_S', 'Pm':'Pm_RN_S', 'Qc':'Qc_RN_S', 'Qg':'Qg_SS_S'}
species_code = species_dict.get(species)

# read csv
df = pd.read_csv('Data/Dataset.csv')[1:60]

# extract data
df = df[['T', 'I', 'D', species_code]]
T = df['T']
I = df['I']
D = df['D']
vn = df[species_code]
vn_max = vn.max()
vn = vn/vn_max

def muf(alpha_log10=1, c_log10=0, g1_log10=1, kxmax_log10=3, p50=-1, ps=-0.5,
        ca=400, Kc=460, q=0.3, R=8.314, Jmax=80, Vcmax=30, z1=0.9, z2=0.9999, a=1.6, L=1):
    alpha, c, g1, kxmax = 10**alpha_log10, 10**c_log10, 10**g1_log10, 10**kxmax_log10
    sapflow_modeled = []
    for i in range(len(vn)):
        # Environmental conditions
        Ti, Ii, Di = T.iloc[i], I.iloc[i], D.iloc[i]
        # px
        pxmin = pxminf(ps, p50)
        if pxmin < ps:
            pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, ps), method='bounded', args=(Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            px1 = pxf2(pxmin, Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            px2 = pxf2(pxmax.x, Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            if px1*px2 < 0:
                px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
                sapflow_modeled.append(kxf(px, kxmax, p50)*(ps-px)*30*60*18/1000000/vn_max/alpha)
            else:
                if abs(px1) < abs(px2):
                    sapflow_modeled.append(1)
                else:
                    sapflow_modeled.append(0)
        else:
            print('pxmin > ps')
            sapflow_modeled.append(0)
    return sapflow_modeled

res = muf()
plt.plot(res)
plt.plot(vn)