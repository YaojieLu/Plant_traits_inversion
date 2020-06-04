import numpy as np
from scipy import optimize
import os

# Soil water potential (Eq. 4) - ps
def psf(s, pe, beta):
    return pe*s**(-beta)

# CO2 compensation point (Eq. A9) - tau
def tauf(T, R):
    return 42.75*np.exp(37830*(T+273.15-298)/(298*R*(T+273.15)))

# Canopy transpiration rate (Eq. 6) - E
def Ef(gs, D, a, L, l, u, n, Z):
    return a*L*l*u/(n*Z)*D*gs

# Whole-tree hydraulic conductance (Eq. 3) - kx
def PLCf(px, p50):
    slope = 16+np.exp(p50)*1092
    f1 = lambda px:1/(1+np.exp(slope/25*(px-p50)))
    PLC = (f1(px) - f1(0))/(1 - f1(0))
    return PLC

def kxf(px, kxmax, p50):
    return kxmax*(1-PLCf(px, p50))

# Photosynthesis rate - A
def Af(gs, T, I,
       Kc, Vcmax, ca, q, Jmax, z1, z2, R):
    tau = tauf(T, R)
    # Eq. A8
    Km = Kc+tau/0.105
    # Rubisco limitation (Eq. A5)
    Ac = 1/2*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tau)*gs+((ca+Km)*gs)**2)**(1/2))
    # Eq. A4
    J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
    # RuBP limitation (Eq. A6)
    Aj = 1/2*(J+(2*tau+ca)*gs-(J**2+2*J*(2*tau-ca+2*tau)*gs+((ca+2*tau)*gs)**2)**(1/2))
    # Am = min(Ac, Aj) Eq. A7
    Am = (Ac+Aj-((Ac+Aj)**2-4*z2*Ac*Aj)**0.5)/(2*z2)
    return Am

# Photosynthesis rate - A
def Atestf(gs, T, I,
           Kc, Vcmax, ca, q, Jmax, z1, z2, R):
    tau = tauf(T, R)
    # Eq. A8
    Km = Kc+tau/0.105
    # Rubisco limitation (Eq. A5)
    Ac = 1/2*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tau)*gs+((ca+Km)*gs)**2)**(1/2))
    # Eq. A4
    J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
    # RuBP limitation (Eq. A6)
    Aj = 1/2*(J+(2*tau+ca)*gs-(J**2+2*J*(2*tau-ca+2*tau)*gs+((ca+2*tau)*gs)**2)**(1/2))
    
    if Ac > Aj:
        return 'Vcmax'
    else:
        return 'J'
# Photosynthesis rate - A
def Atest2f(gs, T, I,
       Kc, Vcmax, ca, q, Jmax, z1, z2, R):
    tau = tauf(T, R)
    # Eq. A8
    Km = Kc+tau/0.105
    # Rubisco limitation (Eq. A5)
    Ac = 1/2*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tau)*gs+((ca+Km)*gs)**2)**(1/2))
    # Eq. A4
    J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
    # RuBP limitation (Eq. A6)
    Aj = 1/2*(J+(2*tau+ca)*gs-(J**2+2*J*(2*tau-ca+2*tau)*gs+((ca+2*tau)*gs)**2)**(1/2))
    # Am = min(Ac, Aj) Eq. A7
    Am = (Ac+Aj-((Ac+Aj)**2-4*z2*Ac*Aj)**0.5)/(2*z2)
    approx = (J/2+Vcmax/2-((J+Vcmax)**2/4-z2*J*Vcmax)**0.5)/(2*z2)
    return Am, approx

# Minimum xylem water potential function at given s
def pxminf(ps, p50):
    f1 = lambda px: -(ps-px)*(1-PLCf(px, p50))
    res = optimize.minimize_scalar(f1, bounds=(ps*1000, ps), method='bounded').x
    return res

def pxf(px,
        T, I, D, ps,
        Kc, Vcmax, ca, q, Jmax, z1, z2, R,
        g1, c,
        kxmax, p50, a, L):
    
    # Plant water balance
    gs = kxf(px, kxmax, p50)*(ps-px)/(1000*a*D*L)
    # PLC modifier
    PLC = PLCf(px, p50)
    f1 = lambda x:np.exp(-x*c)
    # Stomatal function (Eq. 1)
    res = gs-g1*Af(gs, T, I, Kc, Vcmax, ca, q, Jmax, z1, z2, R)/(ca-tauf(T, R))*(f1(PLC)-f1(1))/(f1(0)-f1(1))

    return res

# Medlyn model
def pxf2(px,
         T, I, D, ps,
         Kc, Vcmax, ca, q, Jmax, z1, z2, R,
         g1, c,
         kxmax, p50, a, L):
    
    # Plant water balance
    gs = kxf(px, kxmax, p50)*(ps-px)/(1000*a*D*L)
    # PLC modifier
    PLC = PLCf(px, p50)
    f1 = lambda x:np.exp(-x*c)
    # Stomatal function (Eq. 1)
    res = gs-(1+g1/np.sqrt(D))*Af(gs, T, I, Kc, Vcmax, ca, q, Jmax, z1, z2, R)/ca*(f1(PLC)-f1(1))/(f1(0)-f1(1))

    return res

# Medlyn model 2 with constraint
def pxf3(px,
         T, I, D, ps,
         Kc, Vcmax, ca, q, Jmax, z1, z2, R,
         g1, c,
         kxmax, p50, a, L):
    
    # Martin-StPaul model
    b=(0.3*p50-1)*(np.log(10))**(-1/c)
    # Plant water balance
    gs = kxf(px, kxmax, p50)*(ps-px)/(1000*a*D*L)
    # Stomatal function (Eq. 1)
    res = gs-(1+g1/np.sqrt(D))*Af(gs, T, I, Kc, Vcmax, ca, q, Jmax, z1, z2, R)/ca*np.exp(-(px/b)**c)    
    return res

# Medlyn model 2 without constraint
def pxf4(px,
         T, I, D, ps,
         Kc, Vcmax, ca, q, Jmax, z1, z2, R,
         g1, b, c,
         kxmax, p50, a, L):
    
    # Plant water balance
    gs = kxf(px, kxmax, p50)*(ps-px)/(1000*a*D*L)
    # Stomatal function (Eq. 1)
    res = gs-(1+g1/np.sqrt(D))*Af(gs, T, I, Kc, Vcmax, ca, q, Jmax, z1, z2, R)/ca*np.exp(-(px/b)**c)    
    return res

def vnfsinLAI(LTf, Lamp, Lave, Z, alpha, c, g1, kxmax, p50,
              T, I, Rf, D,
              ca, Kc, q, R, Jmax, Vcmax, z1, z2,
              a, l, u, n, pe, beta, intercept, L0, s0):
        
    s = np.zeros(len(T))
    ps_modeled = np.zeros(len(T))
    sapflow_modeled = np.zeros(len(T))
    
    for i in range(len(T)):
        
        Ti, Ii, Rfi, Di = T[i], I[i], Rf[i] * intercept, D[i]
        if i == 0:
            sp = s0
        else:
            sp = s[i-1]
        Li = Lamp*np.sin(2*np.pi*(i*LTf-L0*LTf+0.75))+Lave
        
        # px & gs
        ps = psf(sp, pe, beta)
        pxmin = pxminf(ps, p50)
        pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, ps), method='bounded', args=(Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
        try:
            px = optimize.brentq(pxf, pxmin, pxmax.x, args=(Ti, Ii, Di, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, Li))
        except ValueError:
            px = -10
            #print('error')
        gs = 10**(-3)*kxf(px, kxmax, p50)*(ps-px)/(a*Di*Li)

        # Soil water balance - s(t) = min(s(t-1) - E(t) + R(t), 1)
        E = Ef(gs, Di, a, Li, l, u, n, Z)
        s[i] = min(sp - E + Rfi/1000/n/Z, 1)
        ps_modeled[i] = ps
        sapflow_modeled[i] = E/alpha
    return ps_modeled, sapflow_modeled

def ensure_dir(file_name):
    directory = 'MCMC_outputs/'+file_name
    directory1 = directory+'1'
    if os.path.exists(directory):
        os.makedirs(directory1)
        os.chdir(directory1)
    else:
        os.makedirs(directory)
        os.chdir(directory)
