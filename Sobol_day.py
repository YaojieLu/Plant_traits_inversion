# Library
from SALib.sample import saltelli
from SALib.analyze import sobol
import xlrd
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf

# Get environmental data
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
T = get_data('T')
I = get_data('I')
D = get_data('D')

# Sobol parameters
Days = range(0, len(T))
#parsSA = ['alpha', 'c', 'g1', 'kxmax', 'p50', 'L', 'ps']
#problem = {'num_vars': len(parsSA),
#           'names': parsSA,
#           'bounds': [[0.001, 0.2],
#                      [2, 20],
#                      [10, 100],
#                      [1, 10],
#                      [-9, -1],
#                      [0.5, 3],
#                      [-4, -1]]}
#                      #[-0.5, -0.1]]}
parsSA = ['c', 'p50', 'L', 'ps']
problem = {'num_vars': len(parsSA),
           'names': parsSA,
           'bounds': [[2, 20],
                      [-9, -1],
                      [0.5, 3],
                      [-4, -1]]}
                      #[-0.5, -0.1]]}
par1 = saltelli.sample(problem, 10000)

# Functions
# Find domain
def testf(X, Env,
          ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999,
          a = 1.6, g1 = 50, kxmax = 7):
    #alpha, c, g1, kxmax, p50, L, ps = X
    c, p50, L, ps = X
    T, I, D = Env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    res = pxf(pxmin, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)*pxf(pxmax.x, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
    return res
# Model
def muf(X, Env,
        ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999,
        a = 1.6, alpha = 0.02, g1 = 50, kxmax = 7):
    #alpha, c, g1, kxmax, p50, L, ps = X
    c, p50, L, ps = X
    T, I, D = Env
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    px = optimize.brentq(pxf, pxmin, pxmax.x, args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    res = alpha*kxf(px, kxmax, p50)*(ps-px)
    return res

# Sobol analysis
ST = []
S1 = []
for i, day in enumerate(Days):
    # Functions
    def testf1(X):
        return testf(X, (T[day], I[day], D[day]))
    def muf1(X):
        return muf(X, (T[day], I[day], D[day]))
    
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
    print(day, " ", Si['ST'])

dfST = pd.DataFrame(ST, columns=parsSA, index=list(Days))
dfS1 = pd.DataFrame(S1, columns=parsSA, index=list(Days))
dfST.index.name='Days'
dfS1.index.name='Days'

# Save to CSV
dfST.to_csv("Results/Sobol_day_4.txt", encoding='utf-8', index=True)
