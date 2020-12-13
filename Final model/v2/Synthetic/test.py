import numpy as np
import matplotlib.pyplot as plt

mean = 50
std = 2
f = lambda x: x*(1/np.sqrt(2*np.pi)/std*np.exp(-0.5*((x-mean)/std)**2))**5

x = np.linspace(50, 150, 250)
y = f(x)
y = y/sum(y)
print(y[:10])
y = np.log10(np.sort(y))[::-1]
plt.plot(y)
print(y[:10])
#plt.plot(y)