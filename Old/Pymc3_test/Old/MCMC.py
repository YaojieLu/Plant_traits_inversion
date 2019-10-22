import xlrd
import numpy as np
from scipy import optimize
import pymc3 as pm3
import theano
import theano.tensor as tt
import matplotlib.pyplot as plt
import pickle

# Read xlsx file
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
species = 'Pm_RN_S'#'Pm_RN_S' 'Am_RN_S' 'Nd_RN_S' 'Qc_RN_S' 'Qg_SN_S'

# Get data from column with specified colname
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:61])
T = get_data('T')
I = get_data('I')
Rf = get_data('Rf')
D = get_data('D')
vn = get_data(species)

model = pm3.Model()
with model:
    ''' Priors '''
    s0 = pm3.Uniform('s0', lower = 0.3, upper = 1, value = 0.5)
    p50 = pm3.Uniform('p50', lower = -10, upper = -0.1, value = -5)
    c = pm3.Uniform('c', lower = -5, upper = -0.1, value = -2)
    Z = pm3.Uniform('Z', lower = 0.5, upper = 5, value = 3)
    alpha = pm3.Uniform('alpha', lower = 0.001, upper = 0.2, value = 0.07)
    bs = pm3.Uniform('bs', lower = 0.1, upper = 2, value = 0.75)
    kxmax = pm3.Uniform('kxmax', lower = 0.5, upper = 20, value = 4.5)
    g1 = pm3.Uniform('g1', lower = 1, upper = 200, value = 20)
    sigma = pm3.HalfNormal('sigma', tau=1)
    
    ''' Other parameters '''
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
    
    ''' Functions '''
    ### PLC
    def PLCf(px,
             p50):
        slope = 16+np.exp(p50)*1092
        f1 = lambda px:1/(1+np.exp(slope/25*(px-p50)))
        PLC = (f1(px) - f1(0))/(1 - f1(0))
        return PLC
    
    ### Xylem conductance
    def kxf(px,
            kxmax, p50):
        return kxmax*(1-PLCf(px, p50))
    
    ### CO2 compensation point
    def tauf(T,
             R = R):
        return 42.75*np.exp(37830*(T+273.15-298)/(298*R*(T+273.15)))
    
    ### Photosynthetic rate
    # Rubisco limitation (Eq. A5)
    def Acf(gs,
            T,
            Kc = Kc, L = L, Vcmax = Vcmax, ca = ca, R = R):
        # Eq. A8
        Km = Kc+tauf(T, R)/0.105
        # Rubisco limitation (Eq. A5)
        Ac = 1/2*L*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tauf(T, R))*gs+((ca+Km)*gs)**2)**(1/2))
        return Ac
    # RuBP limitation (Eq. A6)
    def Ajf(gs,
            I,
            L = L, ca = ca, q = q, Jmax = Jmax, z1 = z1, R = R):
         # Eq. A4
        J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
        # RuBP limitation (Eq. A6)
        Aj = 1/2*L*(J+(2*tauf(T, R)+ca)*gs-(J**2+2*J*(2*tauf(T, R)-ca+2*tauf(T, R))*gs+((ca+2*tauf(T, R))*gs)**2)**(1/2))
        return Aj
    # A = min(Ac, Aj)
    def Af(gs,
           T, I,
           Kc = Kc, L = L, Vcmax = Vcmax, ca = ca, q = q, Jmax = Jmax, z1 = z1, z2 = z2, R = R):
        Ac = Acf(gs, T, Kc, L, Vcmax, ca, R)
        Aj = Ajf(gs, I, L, ca, Jmax)
        # Am = min(Ac, Aj) Eq. A7
        Am = (Ac+Aj-((Ac+Aj)**2-4*z2*Ac*Aj)**0.5)/(2*z2)
        return Am

    ### Xylem water potential
    # Custom theano function - find the root
    def modelf(px,
               T, I, D, ps,
               g1 = g1, c = c, bs = bs, kxmax = kxmax, p50 = p50,
               a = a, l = l, u = u, n = n, Z = Z,
               Kc = Kc, L = L, Vcmax = Vcmax, ca = ca, q = q, Jmax = Jmax, z1 = z1, z2 = z2, R = R):
        # Plant water balance
        gs = 10**(-3)*L*l*u/(n*Z)*kxf(px, kxmax, p50)*(ps-px)/(a*L*l*u/(n*Z)*D)
        # PLC modifier
        PLC = PLCf(px, p50)
        f1 = lambda x:np.exp(-(x/c)**bs)
        # Stomatal function
        value = gs-g1*Af(gs, T, I, Kc, L, Vcmax, ca, q, Jmax, z1, z2, R)/(ca-tauf(T, R))*(f1(PLC)-f1(1))/(f1(0)-f1(1))
        return value

    def jac(px,
            T, I, D, ps,
            Kc, L, Vcmax, ca, q, z1, Jmax, z2,
            g1, c, bs, R,
            kxmax, p50, a, l, u, n, Z):
        kx = kxf(px, kxmax, p50)
        slope = 16+np.exp(p50)*1092
        dkxdpx = (np.exp((1/25)*(-p50+px)*slope)*(1+np.exp((p50*slope)/25))*kxmax*slope)/(25*(1+np.exp((1/25)*(-p50+px)*slope))**2)
        gs = 10**(-3)*L*l*u/(n*Z)*kxf(px, kxmax, p50)*(ps-px)/(a*L*l*u/(n*Z)*D)
        dgsdpx = (-kx+(ps-px)*dkxdpx)/(1000*a*D)
        tau = tauf(T, R)
        Km = Kc + tau/0.105
        J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
        Ac = Acf(gs, T, Kc, L, Vcmax, ca, R)
        Aj = Ajf(gs, I, L, ca, Jmax)
        A = Af(gs, T, I, Kc, L, Vcmax, ca, q, Jmax, z1, z2, R)
        dAcdpx = (1/2)*L*(ca+Km-((-ca+Km+2*tau)*Vcmax+(ca+Km)**2*gs)/np.sqrt(Vcmax**2+2*(-ca+Km+2*tau)*Vcmax*gs+(ca+Km) **2*gs**2))*dgsdpx
        dAjdpx = (1/2)*L*(ca+2*tau+(J*(ca-4*tau)-(ca+2*tau)**2*gs)/np.sqrt(J**2-2*J*(ca-4*tau)*gs+(ca+2*tau)**2*gs**2))*dgsdpx
        dAdpx = (dAcdpx+dAjdpx-(0.5*(-4*z2*Aj*dAcdpx-4*z2*Ac*dAjdpx+2*(Ac+Aj)*(dAcdpx+dAjdpx)))/(-4*z2*Ac*Aj+(Ac+Aj)**2)**0.5)/(2*z2)
        value = (bs*g1*(px/c)**bs*A+px*(np.exp(px/c)**bs*(ca-tau)-g1*dAdpx)*dgsdpx)/(np.exp(px/c)**bs*(px*(ca-tau)))
        return value

    def px_from_TIDpsf(T, I, D, ps):
        return optimize.newton(modelf, ps*1.01, fprime=jac, args=(T, I, D, ps, ))
    
    class Pxf(tt.Op):
        __props__ = ()
        
        itypes = [tt.dscalar, tt.dscalar, tt.dscalar, tt.dscalar]
        otypes = [tt.dscalar]
        
        def perform(self, node, inputs, outputs):
            T, I, D, ps = inputs
            px = px_from_TIDpsf(T, I, D, ps)
            outputs[0][0] = np.array(px)
            
        def grad(self, inputs, g):
            T, I, D, ps, = inputs
            px = self(T, I, D, ps)
            # dpx
            slope = 16+tt.exp(p50)*1092
            PLC = (1/(1+tt.exp(slope/25*(px-p50))) - 1/(1+tt.exp(slope/25*(-p50))))/(1 - 1/(1+tt.exp(slope/25*(-p50))))
            kx = kxmax*(1-PLC)
            dkxdpx = (tt.exp((1/25)*(-p50+px)*slope)*(1+tt.exp((p50*slope)/25))*kxmax*slope)/(25*(1+tt.exp((1/25)*(-p50+px)*slope))**2)
            gs = 10**(-3)*L*l*u/(n*Z)*kx*(ps-px)/(a*L*l*u/(n*Z)*D)
            dgsdpx = (-kx+(ps-px)*dkxdpx)/(1000*a*D)
            tau = 42.75*tt.exp(37830*(T+273.15-298)/(298*R*(T+273.15)))
            Km = Kc + tau/0.105
            J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
            Ac = 1/2*L*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tau)*gs+((ca+Km)*gs)**2)**(1/2))
            Aj = 1/2*L*(J+(2*tau+ca)*gs-(J**2+2*J*(2*tau-ca+2*tau)*gs+((ca+2*tau)*gs)**2)**(1/2))
            A = (Ac+Aj-((Ac+Aj)**2-4*z2*Ac*Aj)**0.5)/(2*z2)
            dAcdpx = (1/2)*L*(ca+Km-((-ca+Km+2*tau)*Vcmax+(ca+Km)**2*gs)/tt.sqrt(Vcmax**2+2*(-ca+Km+2*tau)*Vcmax*gs+(ca+Km)**2*gs**2))*dgsdpx
            dAjdpx = (1/2)*L*(ca+2*tau+(J*(ca-4*tau)-(ca+2*tau)**2*gs)/tt.sqrt(J**2-2*J*(ca-4*tau)*gs+(ca+2*tau)**2*gs**2))*dgsdpx
            dAdpx = (dAcdpx+dAjdpx-(0.5*(-4*z2*Aj*dAcdpx-4*z2*Ac*dAjdpx+2*(Ac+Aj)*(dAcdpx+dAjdpx)))/(-4*z2*Ac*Aj+(Ac+Aj)**2)**0.5)/(2*z2)
            dfdpx = (bs*g1*(px/c)**bs*A+px*(tt.exp(px/c)**bs*(ca-tau)-g1*dAdpx)*dgsdpx)/(tt.exp(px/c)**bs*(px*(ca-tau)))
            # dT
            dtaudT = 42.75*tt.exp((18915*(-24.85+T))/(149*R*(273.15+T)))*(-((18915*(-24.85+T))/(149*R*(273.15+T)**2))+18915/(149*R*(273.15+T)))
            dAcdT = -((gs*L*Vcmax*dtaudT)/tt.sqrt(gs^2*(ca+Km)^2+Vcmax^2+2*gs*Vcmax*(-ca+Km+2*tau)))
            dAjdT = gs*L*(1-(ca*gs+2*J+2*gs*tau)/tt.sqrt(J^2-2*gs*J*(ca-4*tau)+gs^2*(ca+2*tau)^2))*dtaudT
            dAdT = (dAcdT+dAjdT-(0.5*(-4*z2*Aj*dAcdT-4*z2*Ac*dAjdT+2*(Ac+Aj)*(dAcdT+dAjdT)))/(-4*z2*Ac*Aj+(Ac+Aj)**2)**0.5)/(2*z2)
            dfdT = -((g1*((ca-tau)*dAdT+A*dtaudT))/(tt.exp(px/c)**bs*(ca-tau)**2))
            # dI
            dJdI = (q-(0.5*(2*q*(Jmax+I*q)-4*Jmax*q*z1))/((Jmax+I*q)**2-4*I*Jmax*q*z1)**0.5)/(2*z1)
            dAjdI = (1/2)*L*(1+(gs*(ca-4*tau)-J)/tt.sqrt(gs**2*(ca+2*tau)**2-2*gs*(ca-4*tau)*J+J**2))*dJdI
            dAdI = ((1+(Ac*(-1+2*z2)-1*Aj)/(-4*Ac*z2*Aj+(Ac+Aj)**2)**0.5)*dAjdI)/(2*z2)
            dfdI = -((g1*dAdI)/(tt.exp(px/c)**bs*(ca - tau)))
            # dD
            dgsdD = -((kx*(ps - px))/(1000*a*D**2))
            dAcdD = (1/2)*L*(ca+Km-((-ca+Km+2*tau)*Vcmax+(ca+Km)**2*gs)/tt.sqrt(Vcmax**2+2*(-ca+Km+2*tau)*Vcmax*gs+(ca+Km)**2*gs**2))*dgsdD
            dAjdD = (1/2)*L*(ca+2*tau+(J*(ca-4*tau)-(ca+2*tau)**2*gs)/tt.sqrt(J**2-2*J*(ca-4*tau)*gs+(ca+2*tau)**2*gs**2))*dgsdD
            dAdgsD = (dAcdD+dAjdD-(0.5*(-4*z2*Aj*dAcdD-4*z2*Ac*dAjdD+2*(Ac+Aj)*(dAcdD+dAjdD)))/(-4*z2*Ac*Aj+(Ac+Aj)**2)**0.5)/(2*z2)
            dfdD = (1-(g1*dAdgsD)/(tt.exp(px/c)**bs*(ca-tau)))*dgsdD
            # dps
            dgsdps = kx/(1000*a*D)
            dAcdps = (1/2)*L*(ca+Km-((-ca+Km+2*tau)*Vcmax+(ca+Km)**2*gs)/tt.sqrt(Vcmax**2+2*(-ca+Km+2*tau)*Vcmax*gs+(ca+Km)**2*gs**2))*dgsdps
            dAjdps = (1/2)*L*(ca+2*tau+(J*(ca-4*tau)-(ca+2*tau)**2*gs)/tt.sqrt(J**2-2*J*(ca-4*tau)*gs+(ca+2*tau)**2*gs**2))*dgsdps
            dAdgsps = (dAcdps+dAjdps-(0.5*(-4*z2*Aj*dAcdps-4*z2*Ac*dAjdps+2*(Ac+Aj)*(dAcdps+dAjdps)))/(-4*z2*Ac*Aj+(Ac+Aj)**2)**0.5)/(2*z2)
            dfdps = (1-(g1*dAdgsps)/(tt.exp(px/c)**bs*(ca-tau)))*dgsdps
            # Partial derivatives
            dpxdT = -dfdT/dfdpx
            dpxdI = -dfdI/dfdpx
            dpxdD = -dfdD/dfdpx
            dpxdps = -dfdps/dfdpx
            return [g[0]*dpxdT, g[0]*dpxdI, g[0]*dpxdD, g[0]*dpxdps]
    
    # Define Theano variables
    pxtt = tt.dscalar('pxtt')
    Ttt = tt.dscalar('Ttt')
    Itt = tt.dscalar('Itt')
    Dtt = tt.dscalar('Dtt')
    pstt = tt.dscalar('pstt')
    # Define px function
    pxf = theano.function([Ttt, Itt, Dtt, pstt], Pxf()(Ttt, Itt, Dtt, pstt))
    
    ### Soil moisture simulation
    # Define Theano variables
    sE = tt.vector('spE')
    sR = tt.vector('sR')
    ss = tt.dscalar('ss')
    # For loop
    output, updates = theano.scan(fn = lambda E, R, s : tt.minimum(s - E + R, 1),
                                  sequences = [sE, sR],
                                  outputs_info = [ss])
    # Define theano function
    sf = theano.function(inputs = [sE, sR, ss],
                         outputs = output,
                         updates = updates)
    
    ### Transpiration
    
    ''' Simulation '''
    # Soil moisture
    s = sf(vn * alpha, Rf / 1000 / n / Z * intercept, s0)
    px = pxf(T, I, D)
    # Likelihood of observations
    Y_obs = pm3.Normal('Y_obs', mu=muf, tau=sigma, observed=vn)
    
