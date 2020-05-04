
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2

# read MCMC input
df = pd.read_csv('../Data/Dataset.csv')
df = df[['T', 'I', 'D']]
df = df.dropna()

# Sobol parameters
Days = range(len(df))
paras = ['c', 'g1', 'kxmax', 'p50', 'L', 'ps']
problem = {'num_vars': len(paras),
           'names': paras,
#           'bounds': [[0.001, 0.2],
#                      [15, 35],
           'bounds': [[15, 35],
                      [0.1, 1],
                      [1, 10],
                      [-10, -0.5],
                      [0.5, 5],
#                      [-1.5, 0]]}
                      [-4, 0]]}
par1 = saltelli.sample(problem, 2*len(paras)+2)

# model
def muf(X, env, Jmax=100, Vcmax=50,
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999,
        a=1.6, l=1.8*10**(-5), u=48240,
        alpha=0.013):
    c, g1, kxmax, p50, L, ps = X
    T, I, D = env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    px1 = pxf2(pxmin, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
    px2 = pxf2(pxmax.x, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
    if px1*px2 < 0:
        px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
        return l*u*kxf(px, kxmax, p50)*(ps-px)/1000/alpha
    else:
        if abs(px1) < abs(px2):
            return 1
        else:
            return 0

# Sobol analysis
ST = []
S1 = []
for d in Days:
    def muf1(X):
        return muf(X, df[['T', 'I', 'D']].iloc[d, :])
    
    # analysis
    Y = []
    for i, X in enumerate(par1):
        Y.append(muf1(X))
    
    Si = sobol.analyze(problem, np.array(Y))
    ST.append(Si['ST'])
    S1.append(Si['S1'])
    print(d, ' ', Si['ST'])

dfST = pd.DataFrame(ST, columns=paras, index=list(Days))
dfS1 = pd.DataFrame(S1, columns=paras, index=list(Days))
dfST.index.name='Days'
dfS1.index.name='Days'

# Save to CSV
dfST.to_csv('../Results/Sobol_day_CZO.txt', index=True)
