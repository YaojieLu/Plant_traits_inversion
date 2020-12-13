
import numpy as np
from scipy import optimize
import matplotlib.pyplot as plt

# parameters
ps = 0 # soil water potential
px = -1 # end water potential for the whole plant model
p50 = -2 # p50 for the whole-plant model
n = 10 # the number of segments
px_s = np.linspace(ps, px, n+1) # end water potentials for each segment

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
    E_temp=n*(1-PLC)*(px_in-px_out)
    return E-E_temp

# calculation
E=Ef(ps, px, p50)
p50_s = []
for i in range(n):
    try:
        res=optimize.brentq(p50f, 1.001*p50, 0, args=(px_s[i], px_s[i+1], E))
    except:
        res=np.nan
    # print(px_s[i], px_s[i+1])
    # print('p50: {:0.2f} px: {:0.2f}'.format(res, px_s[i+1]))
    p50_s.append(res)

# figure
x=list(range(1, n+1))
fig = plt.figure(figsize=(8, 6))
ax=fig.add_subplot(1, 1, 1)
plt.bar(x, p50_s)
plt.axhline(y=p50, color='r', linestyle='--')
plt.xlim([0, n+1])
#plt.ylim([-3, 0])
plt.xlabel('nth segment', fontsize=20)
plt.ylabel('Segment P50', fontsize=20)
plt.tick_params(labelsize=20)
fig.savefig('../Figures/S4.png', bbox_inches='tight')
