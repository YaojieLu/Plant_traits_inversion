import xlrd
import numpy as np
from scipy import optimize
import pymc3 as pm
import theano.tensor as tt
from theano.compile.ops import as_op

# Read xlsx file
workbook = xlrd.open_workbook('Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
species = 'Pm_RN_S'#'Pm_RN_S' 'Am_RN_S' 'Nd_RN_S' 'Qc_RN_S' 'Qg_SN_S'

# Get data
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:61])
T = tt.as_tensor(get_data('T'))
I = tt.as_tensor(get_data('I'))
Rf = tt.as_tensor(get_data('Rf'))
D = tt.as_tensor(get_data('D'))
vn = tt.as_tensor(get_data(species))

# Define function
@as_op(itypes = [tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,tt.dscalar,
                 tt.dvector,tt.dvector,tt.dvector,tt.dvector], otypes = [tt.dvector])
def vnf(alpha, c, g1, kxmax, p50, s0, Z, T, I, Rf, D,
        ca = 400, R = 8.314, Kc = 460, Vcmax = 30, Jmax = 80, q = 0.3, z1 = 0.9, z2 = 0.9999,
        a = 1.6, L = 1, l = 1.8*10**(-5), u = 48240, n = 0.43,# u = 13.4 hrs
        pe = -2.1*10**(-3), beta = 4.9, intercept = 0.7):
    ### Transpiration rate
    def Evf(T, I, D, s):
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
            tau = tauf(T)
            Km = Kc+tau/0.105
            Ac = 1/2*(Vcmax+(Km+ca)*gs-(Vcmax**2+2*Vcmax*(Km-ca+2*tau)*gs+((ca+Km)*gs)**2)**(1/2))
            return Ac
        # RuBP limitation (Eq. A6)
        def Ajf(gs,
                T, I):
            tau = tauf(T)
            J = (q*I+Jmax-((q*I+Jmax)**2-4*z1*q*I*Jmax)**0.5)/(2*z1)
            Aj = 1/2*(J+(2*tau+ca)*gs-(J**2+2*J*(2*tau-ca+2*tau)*gs+((ca+2*tau)*gs)**2)**(1/2))
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
            gs = kxf(px)*(ps-px)/(1000*a*D)
            # PLC modifier
            PLC = PLCf(px)
            f1 = lambda x:np.exp(-x/c)
            # Empirical model
            value = gs-g1*Af(gs, T, I)/(ca-tauf(T))*(f1(PLC)-f1(1))/(f1(0)-f1(1))
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
        if testmin > 0:
            px = optimize.brenth(modelf, pxmin, pxmax, args=(T, I, D, ps))
            gs = kxf(px)*(ps-px)/(1000*a*D)
            Ev = a*L*l*u/(n*Z)*D*gs
            return Ev
        else:
            return 999
    
    ### Simulation
    s = np.zeros(len(T))
    vn = np.zeros(len(T))
    for i in range(len(T)):
        # Environmental conditions
        Ti, Ii, Rfi, Di = T[i], I[i], Rf[i] * intercept, D[i]
        if i == 0:
            sp = s0
        else:
            sp = s[i-1]
        # Sap flow
        Ev = Evf(Ti, Ii, Di, sp)
        if Ev == 999:
            #print('Bad proposal')
            break
        else:
            s[i] = min(sp - Ev + Rfi/1000/n/Z, 1)
            vn[i] = Ev/alpha
    return vn

with pm.Model() as model:
    ''' Priors '''
    alpha = pm.Uniform('alpha', lower = 0.001, upper = 0.1)
    c = pm.Uniform('c', lower = 0, upper = 0.2)
    g1 = pm.Uniform('g1', lower = 1, upper = 100)
    kxmax = pm.Uniform('kxmax', lower = 0.5, upper = 10)
    p50 = pm.Uniform('p50', lower = -10, upper = -0.1)
    s0 = pm.Uniform('s0', lower = 0.6, upper = 1)
    Z = pm.Uniform('Z', lower = 0.5, upper = 5)
    sigma = pm.HalfNormal('sigma', tau=1)
    
    ''' Sampling '''
    vnmd = vnf(alpha, c, g1, kxmax, p50, s0, Z, T, I, Rf, D) 
    obs = pm.Normal('obs', mu = vnmd, sd = sigma, observed = vn)
    start = {'alpha':0.02, 'c':0.09, 'g1':46.0, 'kxmax':6.2, 'p50':-6.1, 's0':0.7, 'Z':2.9, 'sigma':5.2}
    db = pm.backends.Text(species)
    trace = pm.sample(1e3, step = pm.Metropolis(), start = start, trace = db, chains = 1)#
    pm.traceplot(trace)
