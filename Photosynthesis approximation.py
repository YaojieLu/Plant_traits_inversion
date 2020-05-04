import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import Af

# environmental conditions
T = 21
I = 440
D = 0.012

# functions
def f1(gs):
    return Af(gs, T=T, I=I, Kc=460, Vcmax=31, ca=400, q=0.3, Jmax=48, z1=0.9, z2=0.9999, R=8.314)
def f2(gs, Amax=15, gs50=0.034):
    return Amax*gs/(gs+gs50)

# simulation
x = np.linspace(0, 1, 1000)
y1 = f1(x)
y2 = f2(x)

# figure
fig, axs = plt.subplots(1, 1, figsize=(20, 20))
axs.plot(x, y1, color='black', label='Full model')
axs.plot(x, y2, color='r', label='Simplified version')
plt.legend()