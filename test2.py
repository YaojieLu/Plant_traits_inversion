
import xlrd
import numpy as np
from Functions import Af, Atestf, Atest2f
import matplotlib.pyplot as plt

# Read files
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
# Simulation input
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
T = get_data('T')
I = get_data('I')

# run
for day in range(100):
    print(Atest2f(100, T[day], I[day], ca = 400,
                    Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999))
x = np.linspace(0, 1, 100)
for day in range(100):
    print(Atestf(x[day], T[day], I[day], ca = 400,
                    Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999))
for day in range(100):
    f = lambda gs:Af(gs, T[day], I[day], ca = 400,
                    Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999)
    y = [f(param) for param in x]
    plt.plot(x, y)
