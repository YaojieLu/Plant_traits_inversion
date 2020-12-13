
import pickle
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf4

# species infomation dict
species_dict = {'Aru':['Aru_29', 31, 48], 'Bpa':['Bpa_38', 56, 144],
                'Pgr':['Pgr_22', 61, 122], 'Pst':['Pst_19', 63, 142],
                'Qru':['Qru_42', 51, 88]}

# read MCMC input
df = pd.read_csv('../Data/UMB_daily_average.csv')
#df = df[df['year']==2015]
df['date'] = pd.to_datetime((df['year']*10000+df['month']*100+df['day'])\
                            .apply(str), format='%Y%m%d')

# MCMC model
def muf(X, T, I, D, ps, vn_max, Jmax, Vcmax,\
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, L=1):
    alpha_log10, b, c, g1, kxmax_log10, p50 = X
    alpha, kxmax = 10**alpha_log10, 10**kxmax_log10
    sapflow_modeled = []
    for i in range(len(T)):
        Ti, Ii, Di, psi = T.iloc[i], I.iloc[i], D.iloc[i], ps.iloc[i]
        # px
        pxmin = pxminf(psi, p50)
        pxmax = optimize.minimize_scalar(pxf4, bounds=(pxmin, psi),\
                                         method='bounded',\
                                         args=(Ti, Ii, Di, psi, Kc, Vcmax, ca,\
                                               q, Jmax, z1, z2, R, g1, b, c,\
                                               kxmax, p50, a, L))
        px1 = pxf4(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R,\
                   g1, b, c, kxmax, p50, a, L)
        px2 = pxf4(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2,\
                   R, g1, b, c, kxmax, p50, a, L)
        if px1*px2 < 0:
            px = optimize.brentq(pxf4, pxmin, pxmax.x, args=(Ti, Ii, Di, psi,\
                                                             Kc, Vcmax, ca, q,\
                                                             Jmax, z1, z2, R,\
                                                             g1, b, c, kxmax,\
                                                             p50, a, L))
            sapflow_modeled.append(kxf(px, kxmax, p50)*(psi-px)*30*60*18\
                                   /1000000/vn_max/alpha)
        else:
            if abs(px1) < abs(px2):
                sapflow_modeled.append(1)
            else:
                sapflow_modeled.append(0)
    return sapflow_modeled

# quantiles
def qtf(traces, T, I, D, ps, vn_max, Jmax, Vcmax, n_samples=1000):
    # draw MCMC parameter samples
    tracesdf = pd.DataFrame(data=traces)
    samples = tracesdf.iloc[np.random.choice(tracesdf.index, n_samples)]
    # run
    df_vn = []
    for i in range(len(samples)):
        vn = muf(samples.iloc[i], T, I, D, ps, vn_max, Jmax, Vcmax)
        df_vn.append(vn)
    df_vn = pd.DataFrame(df_vn)
    df_vn_qt = df_vn.quantile([.05, .5, 0.95]).T
    return df_vn_qt

# main function
def f1(species):
    # species info
    species_code, Vcmax, Jmax = species_dict.get(species)
    # read data
    df_sp = df[['date', 'T', 'I', 'D', 'ps15', 'ps30', 'ps60', species_code]]
    df_sp.loc[df_sp[species_code]==0, species_code] = np.nan
    df_sp = df_sp.dropna()
    T = df_sp['T']
    I = df_sp['I']
    D = df_sp['D']
    ps = df[['ps15', 'ps30', 'ps60']].mean(1)
    vn_max = df_sp[species_code].max()
    # read trace
    ts = pickle.load(open('../Data/UMB_trace/Bpa.pickle', 'rb'))
    # run    
    vn = qtf(ts, T, I, D, ps, vn_max, Jmax, Vcmax)
    vn['observed'] = list(df_sp[species_code]/vn_max)
    vn['date'] = list(df_sp['date'])
    vn.rename(columns={0.05: 'qt=0.05', 0.5: 'qt=0.5', 0.95: 'qt=0.95'},\
              inplace=True)
    return vn

# run
vn = f1('Bpa')
vn.to_csv('../Results/ensemble_vn_UMB_Bpa.csv', index=False)
