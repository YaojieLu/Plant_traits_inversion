from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2

# read MCMC input
df = pd.read_csv('../../../Data/UMB_daily_average_Gil_v2.csv')

df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', 'day_len']]
df = df.dropna()

# Sobol parameters
paras = ['alpha_log10', 'c_log10', 'g1_log10', 'kxmax_log10', 'p50', 'ps']
problem = {'num_vars': len(paras),
           'names': paras,
           'bounds': [[-2, 4],
                      [0.7, 1.5],
                      [-1, 5],
                      [-1, 7],
                      [-10, -0.1],
                      [-2, 0]]}
sample_size = 5*(2*len(paras)+2)*1000
par1 = saltelli.sample(problem, sample_size)

# Find domain
def testf(X, env, Vcmax=60, Jmax=120, ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, L=1):
    alpha_log10, c_log10, g1_log10, kxmax_log10, p50, ps = X
    c, g1, kxmax = 10**c_log10, 10**g1_log10, 10**kxmax_log10
    T, I, D, dayi = env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    px1 = pxf2(pxmin, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
    px2 = pxf2(pxmax.x, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
    return px1*px2
# model
def muf(X, env, Vcmax=60, Jmax=120, ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999, a=1.6, L=1):
    alpha_log10, c_log10, g1_log10, kxmax_log10, p50, ps = X
    alpha, c, g1, kxmax = 10**alpha_log10, 10**c_log10, 10**g1_log10, 10**kxmax_log10
    T, I, D, dayi = env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    return kxf(px, kxmax, p50)*(ps-px)*dayi/alpha

# Sobol analysis
ST = []
S1 = []
Days = range(len(df))
for d in Days:    
    # Analysis
    badindex = []
    for j, X in enumerate(par1):
        if testf(X, df[['T', 'I', 'D', 'day_len']].iloc[d, :]) >= 0:
            badindex.append(j)
        if j - len(badindex) >= sample_size:
            break
    
    par2 = np.delete(par1, badindex, 0)
    par2 = par2[0: sample_size]
    
    # analysis
    Y = []
    for X in par2:
        Y.append(muf(X, df[['T', 'I', 'D', 'day_len']].iloc[d, :]))
    
    Si = sobol.analyze(problem, np.array(Y))
    ST.append(Si['ST'])
    S1.append(Si['S1'])
    print(d, ' ', Si['ST'], Si['S1'])

dfST = pd.DataFrame(ST, columns=paras, index=list(Days))
dfS1 = pd.DataFrame(S1, columns=paras, index=list(Days))
dfST.index.name='Days'
dfS1.index.name='Days'

## Save to CSV
#dfS1.to_csv('MCMC_outputs/S1_v2_ps_unk.csv', index=False)
#dfST.to_csv('MCMC_outputs/ST_v2_ps_unk.csv', index=False)
