import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2
#import matplotlib.pyplot as plt
np.random.seed(0)

# read csv
df = pd.read_csv('../../../Data/UMB_daily_average_Gil_v2.csv')

# extract data
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', 'day_len']]
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)
df = df.dropna()
T = df['T']
I = df['I']
D = df['D']
ps = df['ps']
day_len = df['day_len']

def muf(alpha_log10=3, c=13, g1_log10=1.3, kxmax_log10=3, p50=-2.5,
        T=T, I=I, D=D, ps=ps, day_len=day_len,\
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, L=1,
        Jmax=122, Vcmax=61):
    alpha, g1, kxmax = 10**alpha_log10, 10**g1_log10,\
                          10**kxmax_log10
    sapflow_modeled = []
    for i in range(len(T)):
        Ti, Ii, Di, psi = T.iloc[i], I.iloc[i], D.iloc[i], ps.iloc[i]
        dayi = day_len.iloc[i]
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
                                   1000000/alpha*dayi)
        else:
            sapflow_modeled.append(np.nan)
    return sapflow_modeled

baseline = muf()
df_syn = pd.DataFrame()
cols = ['95%', '90%', '85%', '80%', '75%']
errs = [0.0119, 0.02011, 0.02635, 0.03018, 0.03382]
# errs = np.linspace(0, 0.25, 6)
# cols = ['rs='+str(i) for i in errs]
for col, err in zip(cols, errs):
    err = np.random.normal(0, err, len(baseline))
    # err = np.random.normal(0, sum(baseline)/len(baseline)*err, len(baseline))
    df_syn[col] = [max(0, i+j) for i, j in zip(baseline, err)]
    print(round(np.corrcoef(baseline, df_syn[col])[1,0], 4))

# to csv
df_syn[['T', 'I', 'D', 'ps', 'day_len']] = df[['T', 'I', 'D', 'ps', 'day_len']]
df_syn.to_csv('../../../Data/UMB_syn.csv', index=False)
