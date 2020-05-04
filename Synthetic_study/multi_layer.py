import numpy as np
import matplotlib.pyplot as plt

# function
def f(w1, px=-1, ps_1=-0.5, ps_2=-1, p50=-2, kxmax=5):
    slope = 16+np.exp(p50)*1092
    f1 = lambda px:1/(1+np.exp(slope/25*(px-p50)))
    PLC = (f1(px)-f1(0))/(1-f1(0))
    print('PLC = {:0.2f}%'.format(PLC*100))
    E1 = w1*kxmax*(1-PLC)*(ps_1-px)
    E2 = (1-w1)*kxmax*(1-PLC)*(ps_2-px)
    E = E1+E2
    ps = E/(kxmax*(1-PLC))+px
    return ps

# figure
x = np.linspace(0, 1, 11)
y = f(x)
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
plt.plot(x, y)
plt.show()