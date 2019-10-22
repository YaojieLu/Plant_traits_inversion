import numpy as np
from scipy import optimize
import theano
import theano.tensor as tt
theano.config.optdb.max_use_ratio = 20

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

''' Custom theano functions '''
### Xylem water potential
def pxf(bs, c, g1, kxmax, p50,
        T, I, D, s):
    ''' Auxiliary functions '''
    ### Xylem conductance
    # PLC
    def PLCf(px):
        slope = 16+np.exp(p50)*1092
        f1 = lambda px:1/(1+np.exp(slope/25*(px-p50)))
        PLC = (f1(px) - f1(0))/(1 - f1(0))
        return PLC
    # Xylem conductance
    def kxf(px):
        return kxmax*(1-PLCf(px))
    
    ### Photosynthetic rate
    # CO2 compensation point
    def tauf(T):
        return 42.75*np.exp(37830*(T+273.15-298)/(298*R*(T+273.15)))
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
        A = (Ac+Aj-((Ac+Aj)**2-4*z2*Ac*Aj)**0.5)/(2*z2)
        return A

    ### Stomatal model
    def modelf(px,
               T, I, D, ps):
        # Plant water balance
        gs = 10**(-3)*kxf(px)*(ps-px)/(a*D)
        # Empirical model
        value = gs-g1*Af(gs, T, I)/(ca-tauf(T))/(np.exp((px/c)**bs))
        return value
    
    ### Plant's water use envelope
    def pxminf(ps):
        f1 = lambda px: -(ps-px)*kxf(px)
        value = optimize.minimize_scalar(f1, bounds=(ps*1000, ps), method = 'bounded').x
        return value
    
    ''' Root-finding '''
    ps = pe*s**(-beta)
    pxmin = pxminf(ps)
    pxmax = optimize.minimize_scalar(modelf, bounds=(pxmin, ps), method = 'bounded', args=(T, I, D, ps)).x
    testmin = modelf(pxmin, T, I, D, ps)
    testmax = modelf(pxmax, T, I, D, ps)
    #value = optimize.brenth(modelf, pxmin, pxmax, args=(T, I, D, ps))
    #return value
    if testmin > 0 and testmax < 0:
        value = optimize.brenth(modelf, pxmin, pxmax, args=(T, I, D, ps))
        return value
    else:
        return -10.0

class Pxf(tt.Op):
    __props__ = ()
    
    itypes = [tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar]
    otypes = [tt.dscalar]
    
    def perform(self, node, inputs, outputs):
        bs, c, g1, kxmax, p50, T, I, D, s = inputs
        px = pxf(bs, c, g1, kxmax, p50, T, I, D, s)
        outputs[0][0] = np.array(px)
        
    def grad(self, inputs, g):
        bs, c, g1, kxmax, p50, T, I, D, s = inputs
        px = self(bs, c, g1, kxmax, p50, T, I, D, s)
        ps = pe*s**(-beta)
        # dpx
        slope = 16+tt.exp(p50)*1092
        PLC = (1/(1+tt.exp(slope/25*(px-p50))) - 1/(1+tt.exp(slope/25*(-p50))))/(1 - 1/(1+tt.exp(slope/25*(-p50))))
        kx = kxmax*(1-PLC)
        dkxdpx = (tt.exp((1/25)*(-p50+px)*slope)*(1+tt.exp((p50*slope)/25))*kxmax*slope)/(25*(1+tt.exp((1/25)*(-p50+px)*slope))**2)
        gs = 10**(-3)*kx*(ps-px)/(a*D)
        dgsdpx = (-kx+(ps-px)*dkxdpx)/(1000*a*D)
        tau = 42.75*tt.exp(37830*(T+273.15-298)/(298*R*(T+273.15)))
        Km = Kc + tau/0.105
        J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
        Ac = 1/2*L*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tau)*gs+((ca+Km)*gs)**2)**(1/2))
        Aj = 1/2*L*(J+(2*tau+ca)*gs-(J**2+2*J*(2*tau-ca+2*tau)*gs+((ca+2*tau)*gs)**2)**(1/2))
        A = (Ac+Aj-((Ac+Aj)**2-4*z2*Ac*Aj)**0.5)/(2*z2)
        dAcdgs = (1/2)*L*(ca+Km-(2*gs*(ca+Km)**2+2*(-ca+Km+2*tau)*Vcmax)/(2*tt.sqrt(gs**2*(ca+Km)**2+2*gs*(-ca+Km+2*tau)*Vcmax+Vcmax**2)))
        dAjdgs = (1/2)*L*(ca+2*tau+((-ca**2)*gs+ca*(J-4*gs*tau)-4*tau*(J+gs*tau))/tt.sqrt(J**2-2*gs*J*(ca-4*tau)+gs**2*(ca+2*tau)**2))
        dAdgs = (dAcdgs+dAjdgs-(0.5*(-4*z2*Aj*dAcdgs-4*z2*Ac*dAjdgs+2*(Ac+Aj)*(dAcdgs+dAjdgs)))/(-4*z2*Ac*Aj+(Ac+Aj)**2)**0.5)/(2*z2)
        dfdpx = (bs*g1*(px/c)**bs*A+px*(tt.exp((px/c)**bs)*(ca-tau)-g1*dAdgs)*dgsdpx)/(tt.exp((px/c)**bs)*(px*(ca-tau)))
        # dbs
        dfdbs = (A*g1*(px/c)**bs*tt.log(px/c))/(tt.exp((px/c)**bs)*(ca - tau))
        # dc
        dfdc = -((A*bs*g1*(px/c)**bs)/(tt.exp((px/c)**bs)*(c*ca - c*tau)))
        # dg1
        dfdg1 = -(A/(tt.exp((px/c)**bs)*(ca - tau)))
        # dkxmax
        dgsdkxmax = ((ps - px)*(1-PLC))/(1000*a*D)
        dfdkxmax = (1-(g1*dAdgs)/(tt.exp((px/c)**bs)*(ca-tau)))*dgsdkxmax
        # dp50
        dslopedp50 = tt.exp(p50)*1092
        dkxdp50 = (tt.exp((1/25)*(-p50+px)*slope)*kxmax*((-1+tt.exp((1/25)*px*slope))*slope+((-1+tt.exp((1/25)*px*slope))*p50+(1+tt.exp((1/25)*p50*slope))*px)*dslopedp50))/(25*(1+tt.exp((1/25)*(-p50+px)*slope))**2)
        dgsdp50 = ((ps - px)*dkxdp50)/(1000*a*D)
        dfdp50 = (1-(g1*dAdgs)/(tt.exp((px/c)**bs)*(ca-tau)))*dgsdp50
        # dT
        dtaudT = (42.75*37830*tt.exp((37830*(273.15-298+T))/(298*R*(273.15+T))))/(R*(273.15+T)**2)
        dKmdT = dtaudT/0.105
        dAcdT = (1/2)*gs*L*(dKmdT-((ca*gs+Vcmax+gs*Km)*dKmdT+2*Vcmax*dtaudT)/tt.sqrt(Vcmax**2+gs**2*(ca+Km)**2+2*gs*Vcmax*(-ca+Km+2*tau)))
        dAjdT = gs*L*(1-(ca*gs+2*J+2*gs*tau)/tt.sqrt(J**2-2*gs*J*(ca-4*tau)+gs**2*(ca+2*tau)**2))*dtaudT
        dAdT = (dAcdT+dAjdT-(0.5*(-4*z2*Aj*dAcdT-4*z2*Ac*dAjdT+2*(Ac+Aj)*(dAcdT+dAjdT)))/(-4*z2*Ac*Aj+(Ac+Aj)**2)**0.5)/(2*z2)
        dfdT = -((g1*((ca-tau)*dAdT+A*dtaudT))/(tt.exp((px/c)**bs)*(ca-tau)**2))
        # dI
        dJdI = (q-(0.5*(2*q*(Jmax+I*q)-4*Jmax*q*z1))/((Jmax+I*q)**2-4*I*Jmax*q*z1)**0.5)/(2*z1)
        dAjdI = (1/2)*L*(1+(gs*(ca-4*tau)-J)/tt.sqrt(gs**2*(ca+2*tau)**2-2*gs*(ca-4*tau)*J+J**2))*dJdI
        dAdI = ((1+(Ac*(-1+2*z2)-1*Aj)/(-4*Ac*z2*Aj+(Ac+Aj)**2)**0.5)*dAjdI)/(2*z2)
        dfdI = -((g1*dAdI)/(tt.exp((px/c)**bs)*(ca - tau)))
        # dD
        dgsdD = -((kx*(ps - px))/(1000*a*D**2))
        dfdD = (1-(g1*dAdgs)/(tt.exp((px/c)**bs)*(ca-tau)))*dgsdD
        # ds
        dpsds = -beta*pe*s**(-beta-1)
        dgsds = (kx*dpsds)/(1000*a*D)
        dfds = (1-(g1*dAdgs)/(tt.exp((px/c)**bs)*(ca-tau)))*dgsds
        # Implicit function theorm
        dpxdbs = -dfdbs/dfdpx
        dpxdc = -dfdc/dfdpx
        dpxdg1 = -dfdg1/dfdpx
        dpxdkxmax = -dfdkxmax/dfdpx
        dpxdp50 = -dfdp50/dfdpx
        dpxdT = -dfdT/dfdpx
        dpxdI = -dfdI/dfdpx
        dpxdD = -dfdD/dfdpx
        dpxds = -dfds/dfdpx
        return [g[0]*dpxdbs, g[0]*dpxdc, g[0]*dpxdg1, g[0]*dpxdkxmax, g[0]*dpxdp50, g[0]*dpxdT, g[0]*dpxdI, g[0]*dpxdD, g[0]*dpxds]

tt.verify_grad(Pxf(), [0.6, -1., 30., 4.5, -3., 20., 300., 0.01, 1.0], rng=np.random)
