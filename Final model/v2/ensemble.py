import pickle
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2, PLCf
import os

# species infomation dict
species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
species_dict = {'Aru':[31, 48], 'Bpa':[56, 144], 'Pgr':[61, 122],\
                'Pst':[63, 142], 'Qru':[51, 88]}

# read data
df = pd.read_csv('../../Data/UMB_daily_average_Gil_v2.csv')
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)

# MCMC model
def muf(X, T, I, D, ps, Vcmax, Jmax,\
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, rho=997000,\
        L=1):
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
                                         args=(Ti, Ii, Di, psi, Kc, Vcmax,\
                                               ca, q, Jmax, z1, z2, R, g1, c,\
                                               kxmax, p50, a, L))
        px1 = pxf2(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R,\
                   g1, c, kxmax, p50, a, L)
        px2 = pxf2(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2,\
                   R, g1, c, kxmax, p50, a, L)
        if px1*px2 < 0:
            px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(Ti, Ii, Di, psi,\
                                                             Kc, Vcmax, ca,\
                                                             q, Jmax, z1, z2,\
                                                             R, g1, c, kxmax,\
                                                             p50, a, L))
            sapflow_modeled.append(kxf(px, kxmax, p50)*(psi-px)*18/1000000/\
                                   1000*rho/alpha)
            PLC.append(PLCf(px, p50))
            gs_closure.append((f1(PLCf(px, p50))-f1(1))/(f1(0)-f1(1)))
        else:
            sapflow_modeled.append(np.nan)
            PLC.append(np.nan)
            gs_closure.append(np.nan)
    return sapflow_modeled, PLC, gs_closure

# quantiles
def qtf(traces, T, I, D, ps, Vcmax, Jmax, n_samples=1000):
    # draw MCMC parameter samples
    tracesdf = pd.DataFrame(data=traces)
    samples = tracesdf.iloc[np.random.choice(tracesdf.index, n_samples)]
    # run
    df_vn, df_PLC, df_gs = [], [], []
    for i in range(len(samples)):
        vn, PLC, gs = muf(samples.iloc[i], T, I, D, ps, Vcmax, Jmax)
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
def f1(species_code, T, I, D, ps, observed, date, Vcmax, Jmax):
    # read trace
    ts = pickle.load(open('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                          .format(species_code), 'rb'))
    # run    
    vn, PLC, gs = qtf(ts, T, I, D, ps, Vcmax, Jmax)
    vn['observed'] = list(observed)
    vn['date'] = list(date)
    PLC['date'] = list(date)
    gs['date'] = list(date)
    vn.columns = ['qt=0.05', 'qt=0.5', 'qt=0.95', 'observed', 'date']
    PLC.columns = ['qt=0.05', 'qt=0.5', 'qt=0.95', 'date']
    gs.columns = ['qt=0.05', 'qt=0.5', 'qt=0.95', 'date']
    return vn, PLC, gs

# run
for species_code in species_codes:
    if os.path.exists('../../Results/test/ensemble_vn_UMB_Gil_v2_{}.csv'\
                      .format(species_code)):
        print('{} exists'.format(species_code))
        continue
    df_temp = df[['date', 'T', 'I', 'D', 'ps', species_code]]
    #df_temp.loc[df_temp[species_code]==0, species_code] = np.nan
    #df = df.drop(df.index[list(range(76, 79))])
    #df_temp = df_temp.dropna()
    T = df_temp['T']
    I = df_temp['I']
    D = df_temp['D']
    ps = df_temp['ps']
    observed = df_temp[species_code]
    date = df_temp['date']
    
    sp = species_code[:3]
    Vcmax, Jmax = species_dict.get(sp)
    try:
        vn, PLC, gs = f1(species_code, T, I, D, ps, observed, date,\
                         Vcmax, Jmax)
    except FileNotFoundError:
        print('{} not found'.format(species_code))
        continue
    vn.to_csv('../../Results/test/ensemble_vn_UMB_Gil_v2_{}.csv'\
              .format(species_code), index=False)
    PLC.to_csv('../../Results/test/ensemble_PLC_UMB_Gil_v2_{}.csv'\
               .format(species_code), index=False)
    gs.to_csv('../../Results/test/ensemble_gs_UMB_Gil_v2_{}.csv'\
              .format(species_code), index=False)
    print('{} is done'.format(species_code))