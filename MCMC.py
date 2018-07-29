import xlrd
import numpy as np
from scipy import optimize
import pymc3 as pm
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

model = pm.Model()
with model:
    ''' Priors '''
    alpha = pm.Uniform('alpha', lower = 0.001, upper = 0.2)
    bs = pm.Uniform('bs', lower = 0.1, upper = 2)
    c = pm.Uniform('c', lower = -5, upper = -0.1)
    g1 = pm.Uniform('g1', lower = 1, upper = 200)
    kxmax = pm.Uniform('kxmax', lower = 0.5, upper = 20)
    p50 = pm.Uniform('p50', lower = -10, upper = -0.1)
    s0 = pm.Uniform('s0', lower = 0.3, upper = 1)
    sigma = pm.HalfNormal('sigma', tau=1)
    Z = pm.Uniform('Z', lower = 0.5, upper = 5)
    
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
    
    ''' Data '''
    Tod = theano.shared(get_data('T'))
    Iod = theano.shared(get_data('I'))
    Rfod = theano.shared(get_data('Rf'))
    Dod = theano.shared(get_data('D'))
    vnod = theano.shared(get_data(species))
    
    ''' Functions '''
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
        # Eq. A8
        Km = Kc+tauf(T)/0.105
        # Rubisco limitation (Eq. A5)
        Ac = 1/2*L*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tauf(T))*gs+((ca+Km)*gs)**2)**(1/2))
        return Ac
    # RuBP limitation (Eq. A6)
    def Ajf(gs,
            T, I):
         # Eq. A4
        J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
        # RuBP limitation (Eq. A6)
        Aj = 1/2*L*(J+(2*tauf(T)+ca)*gs-(J**2+2*J*(2*tauf(T)-ca+2*tauf(T))*gs+((ca+2*tauf(T))*gs)**2)**(1/2))
        return Aj
    # A = min(Ac, Aj)
    def Af(gs,
           T, I):
        Ac = Acf(gs, T)
        Aj = Ajf(gs, T, I)
        # Am = min(Ac, Aj) Eq. A7
        Am = (Ac+Aj-((Ac+Aj)**2-4*z2*Ac*Aj)**0.5)/(2*z2)
        return Am

    ### Xylem water potential
    # Custom theano function - find the root
    def modelf(px,
               T, I, D, ps):
        # Plant water balance
        gs = 10**(-3)*L*l*u/(n*Z)*kxf(px)*(ps-px)/(a*L*l*u/(n*Z)*D)
        # PLC modifier
        PLC = PLCf(px)
        f1 = lambda x:np.exp(-(x/c)**bs)
        # Stomatal function
        value = gs-g1*Af(gs, T, I)/(ca-tauf(T))*(f1(PLC)-f1(1))/(f1(0)-f1(1))
        return value

    def jac(px,
            T, I, D, ps):
        kx = kxf(px)
        slope = 16+np.exp(p50)*1092
        dkxdpx = (np.exp((1/25)*(-p50+px)*slope)*(1+np.exp((p50*slope)/25))*kxmax*slope)/(25*(1+np.exp((1/25)*(-p50+px)*slope))**2)
        gs = 10**(-3)*L*l*u/(n*Z)*kxf(px)*(ps-px)/(a*L*l*u/(n*Z)*D)
        dgsdpx = (-kx+(ps-px)*dkxdpx)/(1000*a*D)
        tau = tauf(T)
        Km = Kc + tau/0.105
        J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
        Ac = Acf(gs, T)
        Aj = Ajf(gs, T, I)
        A = Af(gs, T, I)
        dAcdpx = (1/2)*L*(ca+Km-((-ca+Km+2*tau)*Vcmax+(ca+Km)**2*gs)/np.sqrt(Vcmax**2+2*(-ca+Km+2*tau)*Vcmax*gs+(ca+Km) **2*gs**2))*dgsdpx
        dAjdpx = (1/2)*L*(ca+2*tau+(J*(ca-4*tau)-(ca+2*tau)**2*gs)/np.sqrt(J**2-2*J*(ca-4*tau)*gs+(ca+2*tau)**2*gs**2))*dgsdpx
        dAdpx = (dAcdpx+dAjdpx-(0.5*(-4*z2*Aj*dAcdpx-4*z2*Ac*dAjdpx+2*(Ac+Aj)*(dAcdpx+dAjdpx)))/(-4*z2*Ac*Aj+(Ac+Aj)**2)**0.5)/(2*z2)
        value = (bs*g1*(px/c)**bs*A+px*(np.exp(px/c)**bs*(ca-tau)-g1*dAdpx)*dgsdpx)/(np.exp(px/c)**bs*(px*(ca-tau)))
        return value
    
    #def pxminf(ps):
    #    f1 = lambda px: -(ps-px)*kxf(px)
    #    value = optimize.minimize_scalar(f1, bounds=(ps*1000, ps), method='bounded').x
    #    return value
    
    def px_from_TIDpsf(T, I, D, ps):
        Len = len(T)
        value = np.zeros(Len)
        for i in range(Len):
        #    pxmin = pxminf(ps[i])       
        #    value[i] = optimize.brentq(modelf, pxmin, ps[i], fprime=jac, args=(T[i], I[i], D[i], ps[i], ))
            value[i] = optimize.newton(modelf, 1.01*ps[i], args=(T[i], I[i], D[i], ps[i], ))
        return value
    
    class Pxf(tt.Op):
        __props__ = ()
        
        itypes = [tt.dvector, tt.dvector, tt.dvector, tt.dvector]
        otypes = [tt.dvector]
        
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
            return [g[0][0]*dpxdT, g[0][0]*dpxdI, g[0][0]*dpxdD, g[0][0]*dpxdps]
            
    ''' Simulation '''
    # Soil moisture simulation
    smd, updates = theano.scan(fn = lambda E, R, s : tt.minimum(s - E + R, 1),
                                  sequences = [vnod * alpha, Rfod / 1000 / n / Z * intercept],
                                  outputs_info = [s0])
    psmd = pe * smd ** (-beta)# Soil water potential
    pxmd = Pxf()(Tod, Iod, Dod, psmd)# Xylem water potential
    slopemd = 16 + tt.exp(p50) * 1092# Slope - xylem vulnerability
    PLCmd = (1/(1+tt.exp(slopemd/25*(pxmd-p50))) - 1/(1+tt.exp(slopemd/25*(-p50))))/(1 - 1/(1+tt.exp(slopemd/25*(-p50))))# PLC
    kxmd = kxmax * (1 - PLCmd)
    vnmd = (kxmd * l * L * (psmd - pxmd) * u) / (1000 * n * Z) * alpha
    
    ''' Sampling '''
    # Likelihood
    obs = pm.Normal('obs', mu = vnmd, sd = sigma, observed = vnod)
    start = pm.find_MAP(fmin = optimize.fmin_powell)
    step = pm.NUTS(scaling = start)
    db = pm.backends.Text('test')
    trace = pm.sample(1e3, step, start = start, trace = db, random_seed = 123)
#map_estimate = pm.find_MAP(model=model, fmin=optimize.fmin_powell)
#print(map_estimate)