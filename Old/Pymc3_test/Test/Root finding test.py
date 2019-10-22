
import numpy as np
from scipy import optimize

ca = 400
Kc = 460
q = 0.3
R = 8.314
Jmax = 80
Vcmax = 30
z1 = 0.9
z2 = 0.9999
a = 1.6
L = 1
l = 1.8*10**(-5)
u = 48240
n = 0.43# u = 13.4 hrs
pe = -2.1*10**(-3)
beta = 4.9
intercept = 0.7

''' Data '''
Tod = np.array([30])
Iod = np.array([200])
Dod = np.array([0.02])
sod = np.array([0.7])

### Xylem water potential
def px_from_TIDpsf(T, I, D, s,
                   bs = 0.9, c = -0.2, g1 = 30, kxmax = 5, p50 = -2, Z = 2):    
    ''' Auxiliary functions '''
    ### PLC
    def PLCf(px):
        slope = 16+np.exp(p50)*1092
        f1 = lambda px:1/(1+np.exp(slope/25*(px-p50)))
        PLC = (f1(px) - f1(0))/(1 - f1(0))
        return PLC
    
    ### Xylem conductance
    def kxf(px):
        return kxmax*(1-PLCf(px))
    
    ### CO2 compensation point
    def tauf(T):
        return 42.75*np.exp(37830*(T+273.15-298)/(298*R*(T+273.15)))
    
    ### Photosynthetic rate
    # Rubisco limitation (Eq. A5)
    def Acf(gs,
            T):
        Km = Kc+tauf(T)/0.105
        Ac = 1/2*L*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tauf(T))*gs+((ca+Km)*gs)**2)**(1/2))
        return Ac
    # RuBP limitation (Eq. A6)
    def Ajf(gs,
            T, I):
        J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
        Aj = 1/2*L*(J+(2*tauf(T)+ca)*gs-(J**2+2*J*(2*tauf(T)-ca+2*tauf(T))*gs+((ca+2*tauf(T))*gs)**2)**(1/2))
        return Aj
    # A = min(Ac, Aj) Eq. A7
    def Af(gs,
           T, I):
        Ac = Acf(gs, T)
        Aj = Ajf(gs, T, I)
        Am = (Ac+Aj-((Ac+Aj)**2-4*z2*Ac*Aj)**0.5)/(2*z2)
        return Am

    ### Xylem water potential
    def modelf(px,
               T, I, D, ps):
        # Plant water balance
        gs = 10**(-3)*kxf(px)*(ps-px)/(a*D)
        # Stomatal model
        value = gs-g1*Af(gs, T, I)/(ca-tauf(T))/(np.exp((px/c)**bs))
        return value
    
    ### Plant's water use envelope
    def pxminf(ps):
        f1 = lambda px: -(ps-px)*kxf(px)
        value = optimize.minimize_scalar(f1, bounds=(ps*1000, ps), method='bounded').x
        return value
    
    ### Root finding
    # Initialization
    Len = len(T)
    value = np.zeros(Len)
    # Soil water potential
    ps = pe*s**(-beta)
    # Iteration
    for i in range(Len):
        pxmin = pxminf(ps[i])
        pxmax = optimize.minimize_scalar(modelf, bounds=(pxmin, ps[i]), method = 'bounded', args=(T[i], I[i], D[i], ps[i])).x
        value[i] = optimize.brenth(modelf, pxmin, pxmax, args=(T[i], I[i], D[i], ps[i]))
    return value

pxest = px_from_TIDpsf(Tod, Iod, Dod, sod)
print(pxest)