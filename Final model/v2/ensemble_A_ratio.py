import pickle
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2, Af

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
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, L=1):
    alpha_log10, c, g1_log10, kxmax_log10, p50 = X
    g1, kxmax = 10**g1_log10, 10**kxmax_log10
    A = []
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
            gs = kxf(px, kxmax, p50)*(psi-px)/(1000*a*Di*L)
            A_gs = Af(gs, Ti, Ii, Kc, Vcmax, ca, q, Jmax, z1, z2, R)
            A_max = Af(1000, Ti, Ii, Kc, Vcmax, ca, q, Jmax, z1, z2, R)
            A.append(A_gs/A_max)
        else:
            A.append(np.nan)
    return A

# quantiles
def qtf(traces, T, I, D, ps, Vcmax, Jmax, n_samples=1000):
    # draw MCMC parameter samples
    tracesdf = pd.DataFrame(data=traces)
    samples = tracesdf.iloc[np.random.choice(tracesdf.index, n_samples)]
    # run
    df_A = []
    for i in range(len(samples)):
        A = muf(samples.iloc[i], T, I, D, ps, Vcmax, Jmax)
        df_A.append(A)
    df_A = pd.DataFrame(df_A)
    df_A_qt = df_A.quantile([.05, .5, 0.95]).T
    return df_A_qt

# main function
def f1(species_code, T, I, D, ps, date, Vcmax, Jmax):
    # read trace
    ts = pickle.load(open('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                          .format(species_code), 'rb'))
    # run    
    A = qtf(ts, T, I, D, ps, Vcmax, Jmax)
    A['date'] = list(date)
    A.columns = ['qt=0.05', 'qt=0.5', 'qt=0.95', 'date']
    return A

# run
for species_code in species_codes:
    # if os.path.exists('../../Results/ensemble_A_UMB_Gil_v2_{}.csv'\
    #                   .format(species_code)):
    #     #print('{} exists'.format(species_code))
    #     continue
    df_temp = df[['date', 'T', 'I', 'D', 'ps']]
    #df_temp.loc[df_temp[species_code]==0, species_code] = np.nan
    #df = df.drop(df.index[list(range(76, 79))])
    #df_temp = df_temp.dropna()
    T = df_temp['T']
    I = df_temp['I']
    D = df_temp['D']
    ps = df_temp['ps']
    date = df_temp['date']
    
    sp = species_code[:3]
    Vcmax, Jmax = species_dict.get(sp)
    try:
        A = f1(species_code, T, I, D, ps, date, Vcmax, Jmax)
    except FileNotFoundError:
        print('{} not found'.format(species_code))
        continue
    A.to_csv('../../Results/ensemble_A_UMB_Gil_v2_{}.csv'\
              .format(species_code), index=False)
    print('{} is done'.format(species_code))