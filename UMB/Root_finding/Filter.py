
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import pxminf, pxf
import matplotlib.pyplot as plt

# read csv
df = pd.read_csv('../Data/UMB_daily_average.csv')
T = df['T']
I = df['I']
D = df['D']
ps = df['ps3']

# parameters
Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, kxmax, a, L = 460, 31, 400, 0.3, 48, 0.9, 0.9999, 8.314, 50, 7, 1.6, 2
c, p50 = 10, -3

j = 0
for i in range(len(T)):
    Ti, Ii, Di, psi = T[i], I[i], D[i], ps[i]
    pxmin = pxminf(psi, p50)
    pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    try:
        px = optimize.brentq(pxf, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    except ValueError:
        j += 1
print('{} out of {} days'.format(j, len(T)))
