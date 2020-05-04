
import pickle
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2

# species infomation dict
species_dict = {'Aru':['Aru_29', 31, 48], 'Bpa':['Bpa_38', 56, 144],
                'Pgr':['Pgr_27', 61, 122], 'Pst':['Pst_2', 63, 142]}

# read MCMC input
df = pd.read_csv('../Data/UMB_daily_average.csv')
df['date'] = pd.to_datetime((df['year']*10000+df['month']*100+df['day']).apply(str), format='%Y%m%d')
df = df[df['year']==2015]
df = df[['date', 'T', 'I', 'D', 'ps15', 'ps30', 'Aru_29', 'Bpa_38', 'Pgr_27', 'Pst_2']]
df = df.dropna()
T = df['T']
I = df['I']
D = df['D']
ps = df[['ps15', 'ps30']].mean(1)

# MCMC model
def muf(X, T, I, D, ps, Jmax, Vcmax,
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999,
        a=1.6, l=1.8*10**(-5), u=48240):
    alpha, c, p50, kxmax, g1, L = X
    sapflow_modeled = []
    for i in range(len(T)):
        Ti, Ii, Di, psi = T.iloc[i], I.iloc[i], D.iloc[i], ps.iloc[i]
        # px
        pxmin = pxminf(psi, p50)
        pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
        px1 = pxf2(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
        px2 = pxf2(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
        if px1*px2 < 0:
            px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            sapflow_modeled.append(l*u*kxf(px, kxmax, p50)*(psi-px)/1000/alpha)
        else:
            if abs(px1) < abs(px2):
                sapflow_modeled.append(1)
            else:
                sapflow_modeled.append(0)
    return sapflow_modeled

# quantiles
def qtf(traces, T, I, D, ps, Jmax, Vcmax, n_samples=1000):
    # draw MCMC parameter samples
    tracesdf = pd.DataFrame(data=traces)
    samples = tracesdf.iloc[np.random.choice(tracesdf.index, n_samples)]
    # run
    df_vn = []
    for i in range(len(samples)):
        vn = muf(samples.iloc[i], T, I, D, ps, Jmax, Vcmax)
        df_vn.append(vn)
    df_vn = pd.DataFrame(df_vn)
    df_vn_qt = df_vn.quantile([.05, .5, 0.95]).T
    return df_vn_qt

# main function 
def f1(species):
    species_code, Vcmax, Jmax = species_dict.get(species)
    # read trace
    ts = pickle.load(open('../Data/UMB_trace/{}.pickle'.format(species), 'rb'))    
    vn = qtf(ts, T, I, D, ps, Jmax, Vcmax)
    vn['observed'] = list(df[species_code])
    vn['date'] = list(df['date'])
    vn.rename(columns={0.05: 'qt=0.05', 0.5: 'qt=0.5', 0.95: 'qt=0.95'}, inplace=True)
    return vn

# run
species_list = ['Aru', 'Bpa', 'Pgr', 'Pst']
for sp in species_list:
    vn = f1(sp)
    vn.to_csv('../Results/ensemble_vn_UMB_{}.csv'.format(sp), index=False)
    print('{} is done'.format(sp))