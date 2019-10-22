
import xlrd
import numpy as np
from scipy import optimize
from Functions import pxminf, kxf, pxf
import matplotlib.pyplot as plt

# Read files
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
# Simulation input
day = 10
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[day])
T = get_data('T')
I = get_data('I')
D = get_data('D')
ps = get_data('Syn_ps_45')

# function
def muf(alpha, kxmax, g1, L,
        c = 16, p50 = -4.5,
        ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999,
        a = 1.6, l = 1.8*10**(-5), u = 48240,):
    # px
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, ps), method='bounded',
                                     args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2,
                                           R, g1, c, kxmax, p50, a, L))
    px = optimize.brentq(pxf, pxmin, pxmax.x, args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2,
                                                    R, g1, c, kxmax, p50, a, L))
    # vn
    vn = l*u*kxf(px, kxmax, p50)*(ps-px)/1000/alpha
    return vn

# run
f = lambda L:muf(alpha=0.02, kxmax=7, g1=50, L=L)
x = np.linspace(0.5, 5, 10)
y = [f(param) for param in x]
plt.plot(x, y)