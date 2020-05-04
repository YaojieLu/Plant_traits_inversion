
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# parameters
ps=0 # soil water potential
dp_s=1 # water potential gradient for segment
dp_w=2 # water potential gradient for whole-plant model
p50_w=-1 # p50 for the whole-plant model

# function
def Ef(px_in, p50, dp):
    px=px_in-dp
    slope=16+np.exp(p50)*1092
    f1=lambda px:1/(1+np.exp(slope/25*(px-p50)))
    PLC=(f1(px)-f1(0))/(1-f1(0))
    #print('PLC={:0.2f}%'.format(PLC*100))
    return (1-PLC)*dp
def p50f(p50, E, px_in, dp):
    px=px_in-dp
    slope=16+np.exp(p50)*1092
    f1=lambda px:1/(1+np.exp(slope/25*(px-p50)))
    PLC=(f1(px)-f1(0))/(1-f1(0))
    E_temp=(1-PLC)*dp
    return E-E_temp
def p50_sf(n): # the nth segment from the root
    E=Ef(ps, p50_w, dp_w)
    px_n=ps-n*dp_s # water potential at the nth segment
    p50=optimize.brentq(p50f, -10, -0.1, args=(E, px_n, dp_s))
    return p50
E = Ef(ps, p50_w, dp_w)
print(E, optimize.brentq(p50f, -10, -0.1, args=(E, 0, 1)))
# # figure
# x=list(range(1, 21))
# y=[p50_sf(n) for n in x]
# fig = plt.figure(figsize = (8, 6))
# ax=fig.add_subplot(1, 1, 1)
# plt.scatter(x, y)
# plt.axhline(y=p50_w, color='r', linestyle='--')
# plt.xlim([0, 21])
# plt.ylim([-3, 0])
# plt.xlabel('nth segment', fontsize=20)
# plt.ylabel('segment p50', fontsize=20)
# plt.tick_params(labelsize=20)
# #plt.tight_layout
