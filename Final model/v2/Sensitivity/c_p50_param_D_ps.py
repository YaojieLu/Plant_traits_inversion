from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import PLCf, pxminf, Af
import matplotlib.pyplot as plt

# read data
df = pd.read_csv('../../../Data/UMB_daily_average_Gil_v2.csv')
df = df[['T', 'I', 'day_len']]
df = df.dropna()

# model
def pxf(px,
        T, I, D, ps,
        Kc, Vcmax, ca, q, Jmax, z1, z2, R,
        param, c,
        p50, a, L):
    
    # Plant water balance
    gs = (1-PLCf(px, p50))*(ps-px)/(1000*a*D*L)
    # PLC modifier
    PLC = PLCf(px, p50)
    f1 = lambda x:np.exp(-x*c)
    # A light-limited
    A_max = Af(1000, T, I, Kc, Vcmax, ca, q, Jmax, z1, z2, R)
    # Stomatal function (Eq. 1)
    res = gs*param-1/np.sqrt(D)*A_max/ca*(f1(PLC)-f1(1))/(f1(0)-f1(1))
    
    return res

def muf(X, env,
        Vcmax=60, Jmax=120, ca=400, Kc=460, q=0.3, R=8.314,
        z1=0.9, z2=0.9999, a=1.6, L=1):
    c, p50, param, D, ps = X
    T, I, day = env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, ps),\
                                     method='bounded',\
                                     args=(T, I, D, ps, Kc, Vcmax, ca, q,\
                                           Jmax, z1, z2, R, param, c, p50,\
                                           a, L))
    px1 = pxf(pxmin, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, param,\
              c, p50, a, L)
    px2 = pxf(pxmax.x, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R,\
              param, c, p50, a, L)
    if px1*px2 < 0:        
        px = optimize.brentq(pxf, pxmin, pxmax.x,\
                             args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1,\
                                   z2, R, param, c, p50, a, L))
        return (1-PLCf(px, p50))*(ps-px)*day
    else:
        return 0

# Sobol analysis
# Sobol parameters
params = ['c', 'p50', 'param', 'D', 'ps']
problem = {'num_vars': len(params),
           'names': params,
           'bounds': [[1, 30],
                      [-2.8, -0.4],
                      [2, 25],
                      [0.002, 0.032],
                      [-1.4, 0]]}
samples = saltelli.sample(problem, 5000)
# analysis
ST = []
S1 = []
for d in range(len(df)):
    Y = []
    for X in samples:
        Y.append(muf(X, df.iloc[d]))
    Si = sobol.analyze(problem, np.array(Y))
    ST.append(Si['ST'])
    S1.append(Si['S1'])
    print(d, ' ', Si['ST'], Si['S1'])
