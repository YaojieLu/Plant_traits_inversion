
import xlrd
import pickle
import numpy as np
import pandas as pd
from Simulation_models import *

# read files
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
ts_Am = pickle.load(open("Data/Am.pickle", "rb"))
ts_Pm = pickle.load(open("Data/Pm.pickle", "rb"))

# simulation input
keys = np.asarray(list(sheet.row_values(0)), dtype = 'str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
T = get_data('T')
I = get_data('I')
Rf = get_data('Rf')
D = get_data('D')

# quantiles
def qtf(traces):
    # draw MCMC samples
    tracesdf = pd.DataFrame(data = traces)
    samples = tracesdf.iloc[np.random.choice(tracesdf.index, 1000)]
    
    # run
    df_vn = []
    df_ps = []
    df_px = []
    for i in range(len(samples)):
        vn, ps, px = vnfsinLAI(samples.iloc[i][:-1], T, I, Rf, D)
        df_vn.append(vn)
        df_ps.append(ps)
        df_px.append(px)
    df_vn = pd.DataFrame(df_vn)
    df_ps = pd.DataFrame(df_ps)
    df_px = pd.DataFrame(df_px)
    df_vn_qt = df_vn.quantile([.05, .5, 0.95]).T
    df_ps_qt = df_ps.quantile([.05, .5, 0.95]).T
    df_px_qt = df_px.quantile([.05, .5, 0.95]).T
    
    return df_vn_qt, df_ps_qt, df_px_qt

df_vn_Am, df_ps_Am, df_px_Am = qtf(ts_Am)
df_vn_Pm, df_ps_Pm, df_px_Pm = qtf(ts_Pm)
df_vn_Am.rename(columns = {0.05: 'qt = 0.05', 0.5: 'qt = 0.5', 0.95: 'qt = 0.95'}, inplace = True)
df_vn_Pm.rename(columns = {0.05: 'qt = 0.05', 0.5: 'qt = 0.5', 0.95: 'qt = 0.95'}, inplace = True)
df_ps_Am.rename(columns = {0.05: 'qt = 0.05', 0.5: 'qt = 0.5', 0.95: 'qt = 0.95'}, inplace = True)
df_ps_Pm.rename(columns = {0.05: 'qt = 0.05', 0.5: 'qt = 0.5', 0.95: 'qt = 0.95'}, inplace = True)
df_px_Am.rename(columns = {0.05: 'qt = 0.05', 0.5: 'qt = 0.5', 0.95: 'qt = 0.95'}, inplace = True)
df_px_Pm.rename(columns = {0.05: 'qt = 0.05', 0.5: 'qt = 0.5', 0.95: 'qt = 0.95'}, inplace = True)
df_vn_Am.to_csv("Results/ensemble_vn_Am.csv", sep='\t', encoding='utf-8')
df_vn_Pm.to_csv("Results/ensemble_vn_Pm.csv", sep='\t', encoding='utf-8')
df_ps_Am.to_csv("Results/ensemble_ps_Am.csv", sep='\t', encoding='utf-8')
df_ps_Pm.to_csv("Results/ensemble_ps_Pm.csv", sep='\t', encoding='utf-8')
df_px_Am.to_csv("Results/ensemble_px_Am.csv", sep='\t', encoding='utf-8')
df_px_Pm.to_csv("Results/ensemble_px_Pm.csv", sep='\t', encoding='utf-8')

# figure
fig = df_ps_Am.plot(figsize = (8, 6))
