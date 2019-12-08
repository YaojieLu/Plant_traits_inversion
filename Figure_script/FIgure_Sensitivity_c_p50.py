
import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf
import matplotlib.pyplot as plt
import seaborn as sns

# model
def muf(c, p50,
        T=12, I=140, D=0.0027, ps=-0.5,
        ca=400, Kc=460, q=0.3, R=8.314, Jmax=80, Vcmax=30, z1=0.9, z2=0.9999,
        a=1.6, l=1.8*10**(-5), u=48240, n=0.43,
        Z=3,
        alpha=0.02, kxmax=7, g1=50, L=2):
    
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf, bounds=(pxmin, ps), method='bounded', args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    px = optimize.brentq(pxf, pxmin, pxmax.x, args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c, kxmax, p50, a, L))
    gs = 10**(-3)*kxf(px, kxmax, p50)*(ps-px)/(a*D*L)
    E = a*L*l*u/(n*Z)*D*gs
    sapflow_modeled = E/alpha
    return sapflow_modeled


# simulation
param_c = np.linspace(5, 20, 151)
param_p50 = np.linspace(-10, -2, 81)
df = [[c, p50, muf(c, p50)] for c in param_c for p50 in param_p50]
df = pd.DataFrame(df, columns =['c', 'p50', 'vn'])
df = df.pivot(index='c', columns='p50', values='vn')

# figure
sns.set(font_scale=1.3)
fig = plt.figure(figsize=(16, 16))
ax = sns.heatmap(df, cmap='viridis', xticklabels=10, yticklabels=10,
                 cbar_kws={'label': '$\\mathit{v_{n}}$'})
plt.xlabel('$\\psi_{x50}$', fontsize=30)
plt.ylabel('$\\mathit{c}$', fontsize=30)
ax.figure.axes[-1].yaxis.label.set_size(30)
plt.savefig('../Figures/Figure Sensitivity_c_p50.png', bbox_inches = 'tight')
