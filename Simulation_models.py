
import numpy as np
import pandas as pd
from Functions import *
from scipy import optimize

# Sin(LAI)
def vnfsinLAI(X, T, I, Rf, D,
              ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999,
              a = 1.6, l = 1.8*10**(-5), u = 48240, n = 0.43,# u = 13.4 hrs
              pe = -2.1*10**(-3), beta = 4.9, intercept = 0.7, L0 = 90, s0 = 0.3):
    
    LTf, Lamp, Lave, Z, alpha, c, g1, kxmax, p50 = X
        
    s = np.zeros(len(T))
    sapflow_modeled = []
    ps = []
    list_px = []
    
    for i in range(len(T)):
        
        Ti, Ii, Rfi, Di = T[i], I[i], Rf[i] * intercept, D[i]
        if i == 0:
            sp = s0
        else:
            sp = s[i-1]
        Li = Lamp*np.sin(2*np.pi*(i*LTf-L0*LTf+0.75))+Lave
        
        # px & gs
        psi = psf(sp, pe, beta)
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
            px1 = pxf(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li)
            px2 = pxf(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li)
            if px1*px2 < 0:
                px = optimize.brentq(pxf, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
            else:
                #print('Bad proposal')
                sapflow_modeled = [100]*len(T)
                break
            
            # Soil water balance
            E = l*u/(n*Z)*kxf(px, kxmax, p50)*(psi-px)/1000
            s[i] = min(sp - E + Rfi/1000/n/Z, 1)
            sapflow_modeled.append(E/alpha)
            ps.append(psi)
            list_px.append(px)
            print(E/(a*Li*l*u/(n*Z)*Di), psi, E/alpha)
        
        else:
            print('gs = 0')
            s[i] = min(sp + Rfi/1000/n/Z, 1)
            ps.append(psi)
            list_px.append(px)
            sapflow_modeled.append(0)
            
    return sapflow_modeled#, ps, list_px

# constant LAI
def vnfconstLAI(X, T, I, Rf, D,
                ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999,
                a = 1.6, l = 1.8*10**(-5), u = 48240, n = 0.43,# u = 13.4 hrs
                pe = -2.1*10**(-3), beta = 4.9, intercept = 0.7, L0 = 90, s0 = 0.3):
    
    alpha, c, g1, kxmax, L, p50, Z = X
    
    s = np.zeros(len(T))
    sapflow_modeled = []
    ps = []
    list_px = []
    
    for i in range(len(T)):
        
        Ti, Ii, Rfi, Di = T[i], I[i], Rf[i] * intercept, D[i]
        if i == 0:
            sp = s0
        else:
            sp = s[i-1]
        Li = L
        
        # px & gs
        psi = psf(sp, pe, beta)
        pxmin = pxminf(psi, p50)
        if pxmin < psi:
            pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, psi), method='bounded', args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
            px1 = pxf(pxmin, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li)
            px2 = pxf(pxmax.x, Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li)
            if px1*px2 < 0:
                px = optimize.brentq(pxf, pxmin, pxmax.x, args=(Ti, Ii, Di, psi, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
            else:
                print('Bad proposal')
                sapflow_modeled = [100]*len(T)
                break
            
            # Soil water balance
            E = l*u/(n*Z)*kxf(px, kxmax, p50)*(psi-px)/1000
            s[i] = min(sp - E + Rfi/1000/n/Z, 1)
            sapflow_modeled.append(E/alpha)
            ps.append(psi)
            list_px.append(px)
            #print(sapflow_modeled[i], Li, psi, gs)
        
        else:
            print('gs = 0')
            s[i] = min(sp + Rfi/1000/n/Z, 1)
            ps.append(psi)
            list_px.append(px)
            sapflow_modeled.append(0)
            
    return sapflow_modeled
