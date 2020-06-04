
import pandas as pd
import matplotlib.pyplot as plt
from Functions import pxf3, kxf, pxminf
from scipy import optimize

# species
species = 'Aru'
species_dict = {'Aru':['Aru_29', 31, 48], 'Bpa':['Bpa_38', 56, 144], 'Pgr':['Pgr_22', 61, 122], 'Pst':['Pst_19', 63, 142], 'Qru':['Qru_42', 51, 88]}
species_code, Vcmax, Jmax = species_dict.get(species)

# read csv
df = pd.read_csv('../Data/UMB_daily_average_Gil_2015.csv')

# extract data
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'day_len', species_code]]
df = df.dropna()
#df = df.drop(df.index[list(range(76, 83))])
T = df['T']
I = df['I']
D = df['D']
ps = df[['ps15', 'ps30']].mean(1)
day_length = df['day_len']
vn_max = df[species_code].max()
vn = df[species_code]/vn_max

def muf(c, p50, kxmax, g1,
        ca=400, Kc=460, q=0.3, R=8.314, Jmax=Jmax, Vcmax=Vcmax, z1=0.9, z2=0.9999, a=1.6, L=3.9):
    sapflow_modeled = []
    for i in range(len(vn)):
        # Environmental conditions
        Ti, Ii, Di, psi, dli = T.iloc[i], I.iloc[i], D.iloc[i], ps.iloc[i], day_length.iloc[i]
        # px
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf3, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            px1 = pxf3(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            px2 = pxf3(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            if px1*px2 < 0:
                px = optimize.brentq(pxf3, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
                sapflow_modeled.append(kxf(px, kxmax, p50)*(psi-px)*30*60*dli*18/1000000/vn_max)
            else:
                print(i, ' ', c, ' ', p50, ' ', Ti)
                if abs(px1) < abs(px2):
                    sapflow_modeled.append(1)
                else:
                    sapflow_modeled.append(0)
        else:
            print('pxmin > ps')
            sapflow_modeled.append(100)
    return sapflow_modeled

# simulation
c = 10
p50 = -2.974111
kxmax = 1.852892/10
g1 = 12.756095/10
vn_model = muf(c, p50, kxmax, g1)

# figure
plt.plot(vn, label='Aru_29')
plt.plot(vn_model, label='model')
plt.legend()