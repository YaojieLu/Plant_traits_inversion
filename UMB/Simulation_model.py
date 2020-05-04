
from Functions import *

# simulation
def UMBf(X, T, I, D, ps,
         ca=400, Kc=460, q=0.3, R=8.314, Jmax=48, Vcmax=31, z1=0.9, z2=0.9999,
         a=1.6, l=1.8*10**(-5), u=48240):   
    alpha, c, p50, kxmax, g1, L = X
    sapflow_modeled = []
    for i in range(len(T)):
        Ti, Ii, Di, psi = T[i], I[i], D[i], ps[i]
        # px
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            px1 = pxf(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            px2 = pxf(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            if px1*px2 < 0:
                px = optimize.brentq(pxf, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            else:
                if abs(px1) < abs(px2):
                    px = pxmin
                else:
                    px = pxmax.x
            # Soil water balance
            sapflow_modeled.append(l*u*kxf(px, kxmax, p50)*(psi-px)/1000/alpha)        
        else:
            print('pxmin > ps')
            sapflow_modeled.append(0)
    return sapflow_modeled

# Medlyn model
def UMBf2(X, T, I, D, ps,
          ca=400, Kc=460, q=0.3, R=8.314, Jmax=48, Vcmax=31, z1=0.9, z2=0.9999,
          a=1.6, l=1.8*10**(-5), u=48240):   
    alpha, c, p50, kxmax, g1, L = X
    sapflow_modeled = []
    for i in range(len(T)):
        Ti, Ii, Di, psi = T[i], I[i], D[i], ps[i]
        # px
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            px1 = pxf2(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            px2 = pxf2(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L)
            if px1*px2 < 0:
                px = optimize.brentq(pxf2, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            else:
                print(i)
                if abs(px1) < abs(px2):
                    px = pxmin
                else:
                    px = pxmax.x
            # Soil water balance
            sapflow_modeled.append(l*u*kxf(px, kxmax, p50)*(psi-px)/1000/alpha)        
        else:
            print('pxmin > ps')
            sapflow_modeled.append(0)
    return sapflow_modeled

# test
def muf(X, T, I, D, ps,
        ca=400, Kc=460, q=0.3, R=8.314, Jmax=48, Vcmax=31, z1=0.9, z2=0.9999,
        a=1.6, l=1.8*10**(-5), u=48240):
    alpha, c, p50, kxmax, g1, L = X
    sapflow_modeled = []
    for i in range(len(T)):
        Ti, Ii, Di, psi = T[i], I[i], D[i], ps[i]
        # px
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            try:
                px = optimize.brentq(pxf, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
            except ValueError:
                print('Bad proposal')
                break
            # vn
            sapflow_modeled.append(l*u*kxf(px, kxmax, p50)*(psi-px)/1000/alpha)
        else:
            print('pxmin > ps')
            sapflow_modeled.append(0)
        print(sapflow_modeled[-1])
    return sapflow_modeled
