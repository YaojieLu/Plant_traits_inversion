import xlrd
import numpy as np
import pandas as pd
from Simulation_models import *
import matplotlib.pyplot as plt

# Read files
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
# Simulation input
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
T = get_data('T')
I = get_data('I')
Rf = get_data('Rf')
D = get_data('D')

#LTf, Lamp, Lave, Z, alpha, c, g1, kxmax, p50
vn = vnfsinLAI([0.0016, 0.5, 2, 3, 0.02, 16, 50, 7, -4.5], T, I, Rf, D)
