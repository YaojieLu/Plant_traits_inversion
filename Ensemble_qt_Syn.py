
import xlrd
import pickle
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import *

# read files
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
ts = pickle.load(open("Data/Am.pickle", "rb"))

# simulation input
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys==lab)[0][0])[1:])
T = get_data('T')
I = get_data('I')
Rf = get_data('Rf')
D = get_data('D')

# simulation
def muf(params,
        T, I, Rf, D,
        ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999,
        a = 1.6, l = 1.8*10**(-5), u = 48240, n = 0.43,# u = 13.4 hrs
        pe = -2.1*10**(-3), beta = 4.9, intercept = 0.7, s0 = 0.3, Z = 3):
    
        alpha, c, p50, kxmax, g1, L = params
        s = np.zeros(len(T))
        sapflow_modeled = np.zeros(len(T))
        
        for i in range(len(T)):
            
            Ti, Ii, Rfi, Di, Li = T[i], I[i], Rf[i] * intercept, D[i], L
            if i == 0:
                sp = s0
            else:
                sp = s[i-1]
            
            # px & gs
            ps = psf(sp, pe, beta)
            pxmin = pxminf(ps, p50)
            pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, ps), method='bounded', args=(Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
            try:
                px = optimize.brentq(pxf, pxmin, pxmax.x, args=(Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
            except ValueError:
                px = -10
                print(i)
            gs = 10**(-3)*kxf(px, kxmax, p50)*(ps-px)/(a*Di*Li)
            
            # Soil water balance - s(t) = min(s(t-1) - E(t) + R(t), 1)
            E = Ef(gs, Di, a, Li, l, u, n, Z)
            s[i] = min(sp - E + Rfi/1000/n/Z, 1)
            sapflow_modeled[i] = E/alpha
        return sapflow_modeled

# quantiles
def qtf(trace):
    # draw MCMC samples
    tracedf = pd.DataFrame(data=trace)
    samples = tracedf.iloc[np.random.choice(tracedf.index, 1000)]
    
    # run
    df_vn = []
    for i in range(len(samples)):
        vn = muf(samples.iloc[i], T, I, Rf, D)
        df_vn.append(vn)
        print(i)
    df_vn = pd.DataFrame(df_vn)
    df_vn_qt = df_vn.quantile([.05, .5, 0.95]).T
    return df_vn_qt

df_vn = qtf(ts)
df_vn.rename(columns={0.05: 'qt = 0.05', 0.5: 'qt = 0.5', 0.95: 'qt = 0.95'}, inplace=True)
df_vn.to_csv("Results/ensemble_Am.csv", sep='\t', encoding='utf-8')
