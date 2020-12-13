import pickle
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2, PLCf

# species infomation dict
species_dict = {'Aru':['Aru_29', 31, 48], 'Bpa':['Bpa_38', 56, 144],
                'Pgr':['Pgr_27', 61, 122], 'Pst':['Pst_2', 63, 142],
                'Qru':['Qru_10', 51, 88]}

# read MCMC input
df = pd.read_csv('../Data/UMB_daily_average.csv')
df = df[df['year']==2015]
df['date'] = pd.to_datetime((df['year']*10000+df['month']*100+df['day'])\
                            .apply(str), format='%Y%m%d')

# MCMC model
def muf(X, T, I, D, ps, Jmax, Vcmax, g1=1,\
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, L=1):
    alpha_log10, c, g1, kxmax_log10, p50 = X
    alpha, kxmax = 10**alpha_log10, 10**kxmax_log10
    sapflow_modeled, gs_closure, PLC = [], [], []
    f1 = lambda x:np.exp(-x*c)
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
            PLC.append(PLCf(px, p50))
            gs_closure.append((f1(PLCf(px, p50))-f1(1))/(f1(0)-f1(1)))
        else:
            sapflow_modeled.append(np.nan)
            PLC.append(np.nan)
            gs_closure.append(np.nan)
    return sapflow_modeled, PLC, gs_closure

# quantiles
def qtf(traces, T, I, D, ps, Jmax, Vcmax, n_samples=1000):
    # draw MCMC parameter samples
    tracesdf = pd.DataFrame(data=traces)
    samples = tracesdf.iloc[np.random.choice(tracesdf.index, n_samples)]
    # run
    df_vn, df_PLC, df_gs = [], [], []
    for i in range(len(samples)):
        vn, PLC, gs = muf(samples.iloc[i], T, I, D, ps, Jmax, Vcmax)
        df_vn.append(vn)
        df_PLC.append(PLC)
        df_gs.append(gs)
    df_vn = pd.DataFrame(df_vn)
    df_PLC = pd.DataFrame(df_PLC)
    df_gs = pd.DataFrame(df_gs)
    df_vn_qt = df_vn.quantile([.05, .5, 0.95]).T
    df_PLC_qt = df_PLC.quantile([.05, .5, 0.95]).T
    df_gs_qt = df_gs.quantile([.05, .5, 0.95]).T
    return df_vn_qt, df_PLC_qt, df_gs_qt

# main function
def f1(species):
    # species info
    species_code, Vcmax, Jmax = species_dict.get(species)
    # read data
    df_sp = df[['date', 'T', 'I', 'D', 'ps15', 'ps30', 'ps60', species_code]]
    df_sp.loc[df_sp[species_code]==0][species_code] = np.nan
    df_sp = df_sp.dropna()
    df_sp = df_sp.drop(df_sp.index[list(range(76, 79))])
    T = df_sp['T']
    I = df_sp['I']
    D = df_sp['D']
    ps = df_sp[['ps15', 'ps30', 'ps60']].mean(1)
    # read trace
    ts = pickle.load(open('../Data/UMB_trace/{}.pickle'.format(species), 'rb'))
    # run    
    vn, PLC, gs = qtf(ts, T, I, D, ps, Jmax, Vcmax)
    vn['observed'] = list(df_sp[species_code])
    vn['date'] = list(df_sp['date'])
    vn.rename(columns={0.05: 'qt=0.05', 0.5: 'qt=0.5', 0.95: 'qt=0.95'},\
              inplace=True)
    PLC['date'] = list(df_sp['date'])
    PLC.rename(columns={0.05: 'qt=0.05', 0.5: 'qt=0.5', 0.95: 'qt=0.95'},\
              inplace=True)
    gs['date'] = list(df_sp['date'])
    gs.rename(columns={0.05: 'qt=0.05', 0.5: 'qt=0.5', 0.95: 'qt=0.95'},\
              inplace=True)
    return vn, PLC, gs

# run
species_list = ['Aru', 'Bpa', 'Pgr', 'Pst']
for sp in species_list:
    vn, PLC, gs = f1(sp)
    vn.to_csv('../Results/ensemble_vn_UMB_{}.csv'.format(sp), index=False)
    PLC.to_csv('../Results/ensemble_PLC_UMB_{}.csv'.format(sp), index=False)
    gs.to_csv('../Results/ensemble_gs_UMB_{}.csv'.format(sp), index=False)
    print('{} is done'.format(sp))