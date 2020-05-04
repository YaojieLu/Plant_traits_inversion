
import numpy as np

# parameters
ps = 0 # soil water potential
px = -2 # end water potential for the whole plant model
p50 = -1 # p50 for the whole-plant model

# function
def Ef(px_in, px_out, p50):
    slope=16+np.exp(p50)*1092
    f1=lambda px:1/(1+np.exp(slope/25*(px-p50)))
    PLC=(f1(px_out)-f1(0))/(1-f1(0))
    #print('PLC={:0.2f}%'.format(PLC*100))
    return (1-PLC)*(px_in-px_out)
def p50f(p50, px_in, px_out, E):
    slope=16+np.exp(p50)*1092
    f1=lambda px:1/(1+np.exp(slope/25*(px-p50)))
    PLC=(f1(px_out)-f1(0))/(1-f1(0))
    E_temp=(1-PLC)*(px_in-px_out)
    return E-E_temp

# calculation
E=Ef(ps, px, p50)
def f(p50):
    return p50f(p50, 0, -0.2, E)
print(f(-0.01), E)