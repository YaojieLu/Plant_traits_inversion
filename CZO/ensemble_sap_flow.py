import pickle
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2, PLCf

# species infomation dict
species_dict = {'Am':'Am_RN_N', 'Nd':'Nd_RN_S', 'Pm':'Pm_RN_S',\
                'Qc':'Qc_RN_S', 'Qg':'Qg_SS_S'}

# read MCMC input
df = pd.read_csv('../Data/Dataset.csv')
df['date'] = pd.to_datetime((df['Year']*10000+df['Month']*100+df['Day'])\
                            .apply(str), format='%Y%m%d')

# MCMC model
def muf(X, T, I, D,\
        Jmax=80, Vcmax=30,
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, L=1):
    alpha_log10, c_log10, g1_log10, kxmax_log10, p50, ps = X
    alpha, c, g1, kxmax = 10**alpha_log10, 10**c_log10, 10**g1_log10, \
        10**kxmax_log10
    sapflow_modeled, gs_closure, PLC = [], [], []
    pxmin = pxminf(ps, p50)
    f1 = lambda x:np.exp(-x*c)
    for i in range(len(T)):
        Ti, Ii, Di = T.iloc[i], I.iloc[i], D.iloc[i]
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
                                   1000000/alpha)
            PLC.append(PLCf(px, p50))
            gs_closure.append((f1(PLCf(px, p50))-f1(1))/(f1(0)-f1(1)))
        else:
            sapflow_modeled.append(np.nan)
            PLC.append(np.nan)
            gs_closure.append(np.nan)
    return sapflow_modeled, PLC, gs_closure

# quantiles
def qtf(traces, T, I, D, n_samples=1000):
    # draw MCMC parameter samples
    tracesdf = pd.DataFrame(data=traces)
    samples = tracesdf.iloc[np.random.choice(tracesdf.index, n_samples)]
    # run
    df_vn, df_PLC, df_gs = [], [], []
    for i in range(len(samples)):
        vn, PLC, gs = muf(samples.iloc[i], T, I, D)
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
    species_code = species_dict.get(species)
    # read data
    df_sp = df[['date', 'T', 'I', 'D', species_code]]
    df_sp.loc[df_sp[species_code]==0][species_code] = np.nan
    df_sp = df_sp.dropna()
    T = df_sp['T']
    I = df_sp['I']
    D = df_sp['D']
    # read trace
    ts = pickle.load(open('../Data/CZO_trace/{}.pickle'.format(species), 'rb'))
    # run    
    vn, PLC, gs = qtf(ts, T, I, D)
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
species_list = ['Am', 'Nd', 'Pm', 'Qg']
for sp in species_list:
    vn, PLC, gs = f1(sp)
    vn.to_csv('../Results/ensemble_vn_CZO_{}.csv'.format(sp), index=False)
    PLC.to_csv('../Results/ensemble_PLC_CZO_{}.csv'.format(sp), index=False)
    gs.to_csv('../Results/ensemble_gs_CZO_{}.csv'.format(sp), index=False)
    print('{} is done'.format(sp))