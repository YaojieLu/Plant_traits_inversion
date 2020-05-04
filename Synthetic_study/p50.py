
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# parameters
ps = 0
px_whole = -0.1
p50_whole = -2
n = 10

# function
def Ef(px_in, px_out, p50):
    slope=16+np.exp(p50)*1092
    f1=lambda px:1/(1+np.exp(slope/25*(px-p50)))
    PLC=(f1(px_out)-f1(0))/(1-f1(0))
    return (1-PLC)*(px_in-px_out)
def p50f(p50, px_in=-0.09, px_out=-0.1, n=n, ps=ps, px_whole=px_whole, p50_wole=p50_whole):
#def p50f(p50, px_in=-0.28, px_out=-0.42, n=n, ps=ps, px_whole=px_whole, p50_wole=p50_whole):
    E = Ef(ps, px_whole, p50_whole)
    slope=16+np.exp(p50)*1092
    f1=lambda px:1/(1+np.exp(slope/25*(px-p50)))
    PLC=(f1(px_out)-f1(0))/(1-f1(0))
    E_temp=n*(1-PLC)*(px_in-px_out)
    return E-E_temp

# figure
x = np.linspace(-2.1, -1.9, 10)
y = p50f(x)
plt.plot(x, y)
plt.axhline(y=0, color='r', linestyle='--')
plt.axvline(x=optimize.brentq(p50f, -2.1, -1.9), color='r', linestyle='--')