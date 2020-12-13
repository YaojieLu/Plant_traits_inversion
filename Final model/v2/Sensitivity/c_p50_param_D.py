from SALib.sample import saltelli
from SALib.analyze import sobol
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import PLCf, pxminf, Af
import matplotlib.pyplot as plt

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

def muf(X,
        T=20, I=273, ps=-0.1,
        Vcmax=60, Jmax=120, ca=400, Kc=460, q=0.3, R=8.314,
        z1=0.9, z2=0.9999, a=1.6, L=1):
    c, p50, param, D = X
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
        return (1-PLCf(px, p50))*(ps-px)
    else:
        return 0

# Sobol analysis
# Sobol parameters
params = ['c', 'p50', 'param', 'D']
problem = {'num_vars': len(params),
           'names': params,
           'bounds': [[1, 30],
                      [-2.8, -0.4],
                      [2, 25],
                      [0.002, 0.032]]}
par1 = saltelli.sample(problem, 5000)
# analysis
Y = []
for X in par1:
    Y.append(muf(X))
S = sobol.analyze(problem, np.array(Y))
df = pd.DataFrame([S['S1'], S['ST']]).T

# figure
df.index = [r'$c$', '$\\psi_{x50}$', r'$\frac{k_{xmax}}{L \ g_1}$', r'$D$']
df.columns = ['First-order', 'Total-order']
fig, ax = plt.subplots(figsize=(10, 10))
df.plot(kind='bar', rot=0, ax=ax)
ax.tick_params(axis='both', labelsize=30)
ax.set_ylabel("Sobol's sensitivity index", fontsize=30)
ax.set_ylim([0, 1])
ax.legend(fontsize=15)
