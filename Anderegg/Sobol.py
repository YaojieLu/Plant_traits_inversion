
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf4

# read MCMC input
df = pd.read_csv('../Data/UMB_daily_average_Gil_2015.csv')
df = df[['date', 'T', 'I', 'D', 'ps15', 'ps30', 'ps60']]
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)
df = df.dropna()

# Sobol parameters
Days = range(len(df))
paras = ['b', 'c', 'g1', 'kxmax', 'p50', 'L']
problem = {'num_vars': len(paras),
           'names': paras,
           'bounds': [[-3, -0.1],
                      [5, 30],
                      [0.1, 1],
                      [1, 10],
                      [-10, -0.5],
                      [0.5, 4]]}
sample_size = 5*(2*len(paras)+2)*1000
par1 = saltelli.sample(problem, sample_size)

# Find domain
def testf(X, env, Jmax=100, Vcmax=50,
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999,
        a=1.6, l=1.8*10**(-5), u=48240):
    b, c, g1, kxmax, p50, L = X
    T, I, D, ps = env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf4, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, b, c, kxmax, p50, a, L))
    px1 = pxf4(pxmin, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, b, c, kxmax, p50, a, L)
    px2 = pxf4(pxmax.x, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, b, c, kxmax, p50, a, L)
    return px1*px2
# model
def muf(X, env, Jmax=100, Vcmax=50,
        ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999,
        a=1.6, l=1.8*10**(-5), u=48240,
        vn_max=2):
    b, c, g1, kxmax, p50, L = X
    T, I, D, ps = env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf4, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, b, c, kxmax, p50, a, L))
    px = optimize.brentq(pxf4, pxmin, pxmax.x, args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, b, c, kxmax, p50, a, L))
    return l*u*kxf(px, kxmax, p50)*(ps-px)/1000/vn_max

# Sobol analysis
ST = []
S1 = []
for d in Days:
    # Functions
    def testf1(X):
        return testf(X, df[['T', 'I', 'D', 'ps']].iloc[d, :])
    def muf1(X):
        return muf(X, df[['T', 'I', 'D', 'ps']].iloc[d, :])
    
    # Analysis
    badindex = []
    for j, X in enumerate(par1):
        if testf1(X) > 0:
            badindex.append(j)
        if j - len(badindex) >= sample_size:
            break
    
    par2 = np.delete(par1, badindex, 0)
    par2 = par2[0: sample_size]
    
    # analysis
    Y = []
    for i, X in enumerate(par2):
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
dfST.to_csv('../Results/Sobol_day_new_model.txt', index=True)
