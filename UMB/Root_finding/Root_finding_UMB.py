
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import *

# read csv
day = 50
df = pd.read_csv('../Data/UMB_daily_average.csv')
T = df['T'][day]
I = df['I'][day]
D = df['D'][day]
ps = df['ps'][day]

# parameters
Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, kxmax, a, L = 460, 31, 400, 0.3, 48, 0.9, 0.9999, 8.314, 50, 7, 1.6, 2
c, p50 = 20, -1

# environmental conditions
ps_list = [ps, ps*2]*2
# results
fig, axs = plt.subplots(2, 2, figsize=(20, 20))
i = 0
for row in axs:
    for col in row:
        psi = ps_list[i]
        # px function
        def f1(px):
            return pxf(px,
                       T=T, I=I, D=D, ps=psi,
                       Kc=Kc, Vcmax=Vcmax, ca=ca, q=q, Jmax=Jmax, z1=z1, z2=z2, R=R,
                       g1=g1, c=c, kxmax=kxmax, p50=p50, a=a, L=L)
        # LHS
        def f2(px,
               D=D, ps=psi,
               kxmax=kxmax, p50=p50, a=a, L=L):
            gs = kxf(px, kxmax, p50)*(ps-px)/(1000*a*D*L)
            return gs
        # RHS
        def f3(px,
               T=T, I=I, D=D, ps=psi,
               Kc=Kc, Vcmax=Vcmax, ca=ca, q=q, Jmax=Jmax, z1=z1, z2=z2, R=R,
               g1=g1, c=c, kxmax=kxmax, p50=p50, a=a, L=L):
            
            # Plant water balance
            gs = kxf(px, kxmax, p50)*(ps-px)/(1000*a*D*L)
            # PLC modifier
            PLC = PLCf(px, p50)
            f1 = lambda x:np.exp(-x*c)
            # Stomatal function (Eq. 1)
            res = g1*Af(gs, T, I, Kc, Vcmax, ca, q, Jmax, z1, z2, R)/(ca-tauf(T, R))*(f1(PLC)-f1(1))/(f1(0)-f1(1))    
            return res
        # RHS f1
        def f4(px,
               T=T, I=I, D=D, ps=psi,
               Kc=Kc, Vcmax=Vcmax, ca=ca, q=q, Jmax=Jmax, z1=z1, z2=z2, R=R,
               kxmax=kxmax, p50=p50, a=a, L=L):
            # Plant water balance
            gs = kxf(px, kxmax, p50)*(ps-px)/(1000*a*D*L)
            # Stomatal function (Eq. 1)
            res = Af(gs, T, I, Kc, Vcmax, ca, q, Jmax, z1, z2, R)/(ca-tauf(T, R))
            return res
        # RHS f2
        def f5(px,
               c=c, p50=p50):
            # PLC modifier
            PLC = PLCf(px, p50)
            f1 = lambda x:np.exp(-x*c)
            # Stomatal function (Eq. 1)
            res = (f1(PLC)-f1(1))/(f1(0)-f1(1))    
            return res
        # results
        pxmin = pxminf(psi, p50)
        ypxmin = f1(pxmin)
        pxmax = optimize.minimize_scalar(f1, bounds=(pxmin, psi), method='bounded').x
        ypxmax = f1(pxmax)
        x = np.linspace(pxmin*1.1, psi, 100) #px
        if i < 2:
            y1 = f1(x)
            y2 = f2(x)
            y3 = f3(x)
            col.plot(x, y2, color="r")
            col.plot(x, y3, color="b")
            col.scatter(x, y1, linewidth=1, color="black")
            col.set_title('{} = {:0.3f} MPa'.format('$\psi_{s}$', psi), fontsize=30)
#            col.axhline(y=0, color='b', linestyle='-', linewidth=0.5)
#            col.plot(pxmin, ypxmin, marker='o', markersize=10, color="red")
#            col.plot(pxmax, ypxmax, marker='o', markersize=10, color="red")
            col.axes.get_xaxis().set_visible(False)
#            print("pxmin = {:.6f} ypxmin = {:.6f}\npxmax = {:.6f} ypxmax = {:.6f}".format(pxmin, ypxmin, pxmax, ypxmax))
            try:
                px = optimize.brentq(f1, pxmin, pxmax)
                print("px = {:0.6f}".format(px))
#            col.plot(px, 0, marker='o', markersize=10, color="b")
            except ValueError:
                print('Bad proposal')
        else:
            y4 = f4(x)
            y5 = f5(x)
            col.plot(x, y4, linewidth=1, color="black")
            col.plot(x, y5, color="r")
            col.set_xlabel('$\psi_{x}$ (MPa)', fontsize=30)
            col.tick_params(axis='x', labelsize=30)
        if i == 1:
            col.legend(['LHS', 'RHS', 'Difference'], loc='upper left', fontsize=20, framealpha=0)
        elif i ==3:
            col.legend(['f1 (assimilation limited)', 'f2 (water limited)'], loc='upper left', fontsize=20, framealpha=0)
        i += 1
plt.subplots_adjust(wspace=0.1, hspace=0.05)

## stomatal response to drought
#def f(PLC, c=5):
#    f1 = lambda x:np.exp(-x*c)
#    res = (f1(PLC)-f1(1))/(f1(0)-f1(1))
#    return res
#x = np.linspace(0, 1, 100) #PLC
#y = f(x) # stomatal closure
#fig, axs = plt.subplots(1, 1, figsize=(10, 10))
#axs.plot(x, y)
