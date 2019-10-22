
import xlrd
import pickle
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf

# read files
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
ts = pickle.load(open("Data/std=30%.pickle", "rb"))

# simulation input
keys = np.asarray(list(sheet.row_values(0)), dtype = 'str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
T = get_data('T')
I = get_data('I')
D = get_data('D')
ps = get_data('Syn_ps_45')

# MCMC model
def muf(X, T, I, D, ps,
        ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999, a = 1.6, l = 1.8*10**(-5), u = 48240,):
    alpha, c, p50, kxmax, g1, L = X
    sapflow_modeled = np.zeros(len(T))
    for i in range(len(T)):
        
        # Environmental conditions
        Ti, Ii, Di, psi, Li = T[i], I[i], D[i], ps[i], L
        
        # px
        pxmin = pxminf(psi, p50)
        pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
        px = optimize.brentq(pxf, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
        # vn
        sapflow_modeled[i] = l*u*kxf(px, kxmax, p50)*(psi-px)/1000/alpha
    
    return sapflow_modeled

# quantiles
def qtf(traces):
    # draw MCMC samples
    tracesdf = pd.DataFrame(data = traces)
    samples = tracesdf.iloc[np.random.choice(tracesdf.index, 1000)]
    
    # run
    df_vn = []
    for i in range(len(samples)):
        vn = muf(samples.iloc[i], T, I, D, ps)
        df_vn.append(vn)
    df_vn = pd.DataFrame(df_vn)
    df_vn_qt = df_vn.quantile([.05, .5, 0.95]).T
    
    return df_vn_qt

df_vn = qtf(ts)
df_vn.rename(columns = {0.05: 'qt = 0.05', 0.5: 'qt = 0.5', 0.95: 'qt = 0.95'}, inplace = True)
df_vn.to_csv("Results/ensemble_vn_noisy.csv", sep='\t', encoding='utf-8')
