
import xlrd
import numpy as np
import pandas as pd
from Simulation_models import vnfsinLAI
import matplotlib.pyplot as plt

# read xlsx file
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')

# get data from column with specified colname
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys==lab)[0][0])[1:])
T = get_data('T')
I = get_data('I')
Rf = get_data('Rf')
D = get_data('D')
vn_true = get_data('Synthetic_45')
# analysis
def vnf(params):
    c, p50 = params
    alpha, g1, kxmax = 0.02, 50, 7
    LTf, Lamp, Lave, Z = 0.0016, 0.5, 2, 3
    vn = vnfsinLAI([LTf, Lamp, Lave, Z, alpha, c, g1, kxmax, p50], T, I, Rf, D)
    return vn
labels = ['c', 'p50']
#base = [16, -4.5]
#reso = 10
#df = np.array([base, ]*reso*len(labels))
#for i in range(len(labels)):
#    df[i*reso:(i*reso+reso), i] = np.linspace(base[i]/2, base[i]*2, reso)
#vn = np.array([abs(vnf(param)-vn_true) for param in df])
#vn = pd.DataFrame(vn)
#vn = vn.T
#err = vn.sum()
#err2 = err[err<100]
#err2.plot()
c = np.linspace(15, 20, 10)
p50 = np.linspace(-5, -4, 10)
vn = np.array([sum(abs(vnf((c_value, p50_value))-vn_true)) for c_value in c for p50_value in p50])
vn = pd.DataFrame(vn.reshape(10, 10))