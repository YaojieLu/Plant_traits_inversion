import xlrd
import numpy as np
from scipy import optimize
import theano
import theano.tensor as tt

# Read xlsx file
workbook = xlrd.open_workbook('../Data/Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
species = 'Pm_RN_S'#'Pm_RN_S' 'Am_RN_S' 'Nd_RN_S' 'Qc_RN_S' 'Qg_SN_S'

# Get data
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])

''' Priors '''
alpha, bs, c, g1, kxmax, p50, s0, Z = tt.dscalars('alpha', 'bs', 'c', 'g1', 'kxmax', 'p50', 's0', 'Z')

''' Data '''
Tod = get_data('T')
Iod = get_data('I')
Rfod = get_data('Rf')
Dod = get_data('D')
vnod = get_data(species)

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
        print('Bad proposal')
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

''' Simulation '''
# Soil moisture simulation
Estt, Rstt = tt.dvectors('Estt', 'Rstt')
sstt = tt.dscalar('sstt')
smd, updates = theano.scan(fn = lambda E, R, s : tt.minimum(s - E + R, 1),
                           sequences = [Estt, Rstt],
                           outputs_info = [sstt])
sf = theano.function(inputs=[Estt, Rstt, sstt],
                    outputs = smd,
                    updates = updates)
ss = sf(vnod * 0.01, Rfod / 1000 / n / 3 * intercept, 0.8)

# Sap flow
Tvntt, Ivntt, Dvntt, svntt = tt.dvectors('Tvntt', 'Ivntt', 'Dvntt', 'svntt')
def step(T, I, D, s, alpha, bs, c, g1, kxmax, p50, Z):
    ps = pe * s ** (-beta) # Soil water potential
    px = Pxf()(bs, c, g1, kxmax, p50, T, I, D, s) # Xylem water potential
    slope = 16 + tt.exp(p50) * 1092 # Slope - xylem vulnerability
    PLC = (1/(1+tt.exp(slope/25*(px-p50))) - 1/(1+tt.exp(slope/25*(-p50))))/(1 - 1/(1+tt.exp(slope/25*(-p50)))) # PLC
    kx = kxmax * (1 - PLC) # Xylem conductance
    vn = (kx * l * L * (ps - px) * u) / (1000 * n * Z) / alpha # Sap flow
    return vn, px
vnmd, updatevn = theano.scan(fn = step,
                             sequences = [Tvntt, Ivntt, Dvntt, svntt],
                             non_sequences = [alpha, bs, c, g1, kxmax, p50, Z])
vnf = theano.function(inputs=[Tvntt, Ivntt, Dvntt, svntt, alpha, bs, c, g1, kxmax, p50, Z],
                      outputs = vnmd,
                      updates = updatevn)
vmss = vnf(Tod, Iod, Dod, ss, 0.01, 1, -1, 100, 5, -2, 3)
#print(vmss)
#print(ss)