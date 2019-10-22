# Library
from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf

# Get environmental data
T = [12]#[5, 30]
I = [140]#np.linspace(50, 500, 10)
D = [0.0027]#np.linspace(0.0002, 0.03, 10)
Env = [(t, i, d) for t in T for i in I for d in D]

# Sobol parameters
parsSA = ['c', 'kxmax', 'p50', 'L', 'ps']
problem = {'num_vars': len(parsSA),
           'names': parsSA,
#           'bounds': [[0, 0.2],
#                      [2, 10],
#                      [-10, -0.1],
#                      [0.5, 5],
#                      [-5, -0.1]]}
           'bounds': [[2, 20],
                      [2, 10],
                      [-9, -1],
                      [0.5, 3],
                      [-0.5, -0.1]]}
par1 = saltelli.sample(problem, 10000)

# Functions
# Find domain
def testf(X, Env,
          ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999,
          a = 1.6, g1 = 50):
    c, kxmax, p50, L, ps = X
    T, I, D = Env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    res = pxf(pxmin, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)*pxf(pxmax.x, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
    return res
# Model
def muf(X, Env,
        ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999,
        a = 1.6, g1 = 50):
    c, kxmax, p50, L, ps = X
    T, I, D = Env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    px = optimize.brentq(pxf, pxmin, pxmax.x, args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    res = kxf(px, kxmax, p50)*(ps-px)
    return res

# Sobol analysis
ST = []
S1 = []
for day, env in enumerate(Env):
    # Functions
    def testf1(X):
        return testf(X, env)
    def muf1(X):
        return muf(X, env)
    
    # Analysis
    badindex = []
    for j, X in enumerate(par1):
        if testf1(X) > 0:
            badindex.append(j)
        if j - len(badindex) >= (2*len(parsSA)+2)*1000:
            break
    
    par2 = np.delete(par1, badindex, 0)
    par2 = par2[0:(2*len(parsSA)+2)*1000]

    Y = np.zeros([par2.shape[0]])
    for j, X in enumerate(par2):
        Y[j] = muf1(X)
    
    Si = sobol.analyze(problem, Y)
    ST.append(Si['ST'])
    S1.append(Si['S1'])
    print(day, " ", Si['ST'], Si['S1'])

# Save to CSV
df_Env = pd.DataFrame(Env, columns = ['T', 'I', 'D'])
df_ST = pd.DataFrame(ST, columns = parsSA)
df_ST = pd.concat([df_Env, df_ST], axis = 1)
#df_ST.to_csv("Results/Sobol_env.txt", encoding = 'utf-8', index = True)
