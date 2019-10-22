import xlrd
from pymc import Uniform, HalfNormal, Normal, deterministic, MCMC, Matplot, AdaptiveMetropolis
from Functions import *
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import pickle

# Read xlsx file
workbook = xlrd.open_workbook('Dataset.xlsx')
sheet = workbook.sheet_by_name('daily_average')
dictsp = {'Am':'Am_RN_N', 'Nd':'Nd_RN_S', 'Pm':'Pm_RN_S', 'Qc':'Qc_RN_S', 'Qg':'Qg_SS_S', 'Syn':'Synthetic'}
species = 'Am'

# Get data from column with specified colname
keys = np.asarray(list(sheet.row_values(0)), dtype='str')
get_data = lambda lab: np.asarray(sheet.col_values(np.where(keys == lab)[0][0])[1:])
T = get_data('T')
I = get_data('I')
Rf = get_data('Rf')
D = get_data('D')
vn = get_data(dictsp.get(species))

''' Priors ''' 
alpha = Uniform('alpha', lower = 0.001, upper = 0.2, value = 0.02)
c = Uniform('c', lower = 2, upper = 20, value = 16)
g1 = Uniform('g1', lower = 1, upper = 100, value = 50)
kxmax = Uniform('kxmax', lower = 0.5, upper = 10, value = 4.5)
Lamp = Uniform('Lamp', lower = 0, upper = 1, value = 0.5)
Lave = Uniform('Lave', lower = 1, upper = 3, value = 2)
LTf = Uniform('LTf', lower = 1/3650, upper = 1/365, value = 1/1000)
p50 = Uniform('p50', lower = -10, upper = -0.1, value = -5)
Z = Uniform('Z', lower = 0.5, upper = 5, value = 3)
sigma = HalfNormal('sigma', tau = 1)

''' deterministic model ''' 
@deterministic
def muf(alpha = alpha, c = c, g1 = g1, kxmax = kxmax, Lamp = Lamp, Lave = Lave, LTf = LTf, p50 = p50, Z = Z,
        ca = 400, Kc = 460, q = 0.3, R = 8.314, Jmax = 80, Vcmax = 30, z1 = 0.9, z2 = 0.9999,
        a = 1.6, l = 1.8*10**(-5), u = 48240, n = 0.43,# u = 13.4 hrs
        pe = -2.1*10**(-3), beta = 4.9, intercept = 0.7, L0 = 90, s0 = 0.3):
    
    s = np.zeros(len(vn))
    sapflow_modeled = []
    
    for i in range(len(vn)):
        
        # Environmental conditions
        Ti, Ii, Rfi, Di = T[i], I[i], Rf[i] * intercept, D[i]
        
        if i == 0:
            sp = s0
        else:
            sp = s[i-1]
        
        Li = Lamp*np.sin(2*np.pi*(i*LTf-L0*LTf+0.75))+Lave
        
        # px
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
                sapflow_modeled = [100]*len(vn)
                break
            # vn
            E = l*u/(n*Z)*kxf(px, kxmax, p50)*(psi-px)/1000
            s[i] = min(sp - E + Rfi/1000/n/Z, 1)
            sapflow_modeled.append(E/alpha)
        else:
            print('gs = 0')
            s[i] = min(sp + Rfi/1000/n/Z, 1)
            sapflow_modeled.append(0)
    return sapflow_modeled

'''data likelihoods'''
np.random.seed(1)
Y_obs = Normal('Y_obs', mu=muf, tau=sigma, value=vn, observed=True)

''' posterior sampling '''
M = MCMC([alpha, c, g1, kxmax, Lamp, Lave, LTf, p50, Z, sigma])
M.use_step_method(AdaptiveMetropolis, [alpha, c, g1, kxmax, Lamp, Lave, LTf, p50, Z, sigma])
M.sample(iter=1000000, burn=500000, thin=40)

# Save trace
ensure_dir(species)
traces = {'alpha':M.trace('alpha')[:], 'c':M.trace('c')[:], 'g1':M.trace('g1')[:], 'kxmax':M.trace('kxmax')[:], 'Lamp':M.trace('Lamp')[:], 'Lave':M.trace('Lave')[:], 'LTf':M.trace('LTf')[:], 'p50':M.trace('p50')[:], 'Z':M.trace('Z')[:], 'sigma':M.trace('sigma')[:]}
pickle_out = open("MCMC.pickle", "wb")
pickle.dump(traces, pickle_out)
pickle_out.close()

# Trace
Matplot.plot(M)
