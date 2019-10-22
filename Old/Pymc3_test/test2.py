import numpy as np
import matplotlib.pyplot as plt

def simulate_rainfall(n_trajectories, tRun, dt, lam, gam):
    size = tRun*n_trajectories 
    depthExp = -np.log(1.0-np.random.random(size=size))/gam
    freqUnif = np.random.random(size=size)
    
    depth = np.zeros(size)
    # the occurence of rainfall in any independent interval is lam*dt
    depth[freqUnif<np.tile(lam,size)*dt] = depthExp[freqUnif<np.tile(lam,size)*dt] # rain falls according to prob within an increment
    depth_re = np.reshape(depth, (n_trajectories, tRun))
    return depth_re

def simulate_s_t(depths, tRun, dt, s0, Emax, Zr):
    
    s_t = np.zeros(tRun); s0 = s0
    E_t = np.zeros_like(s_t)
    L_t = np.zeros_like(s_t)
    
    for i in range(tRun): 
        R_normed = depths[i]
        Infil_normed = min(R_normed, 1.0-s0)    # from the previous step
        s1 = s0 + Infil_normed                  # update with rainfall 
        
        E = Emax*s1/(n*Zr)                  # evapotranspiration
        L = Ksat*s1**(2*b+3)/(n*Zr)         # leakage
        L = 0                               # anything above 1 is lost to leakage
        # soil moisture water balance
        s_out = max(s1 - dt*(E + L), 10**(-20)) # drawdown 
        
        # update to next step
        s_t[i] = s_out; s0 = s_out
        E_t[i] = E
        L_t[i] = L
    return s_t, E_t, L_t

def simulate_ps_t(n_trajectories, tRun, dt, s0, lam, alpha, Emax, Zr):
    
    gam = (n*Zr)/(alpha)
    # set up repositories 
    depth_re = simulate_rainfall(n_trajectories, tRun, dt, lam, gam)
    ps_samples = np.zeros((n_trajectories, tRun))
    E_samples = np.zeros_like(ps_samples)
    L_samples = np.zeros_like(ps_samples)
    for nsim in range(n_trajectories):
        s_t, E_t, L_t = simulate_s_t(depth_re[nsim], tRun, dt, s0, Emax, Zr)
        ps_samples[nsim] = s_t
        E_samples[nsim] = E_t
        L_samples[nsim] = L_t
    return ps_samples, E_samples, L_samples


## soil parameters 
Ksat = 1.0              # cm/day - for loamy sand
Psat = -0.17*10**(-3)   # MPa - for loamy sand
b = 4.38                # exponent - for loamy sand
n = 0.45                # porosity - for loamy sand

## simulation parameters
n_trajectories = 100
dt = 0.1
tRun = int(365/dt)
lam = 0.20      # 1/day
alpha = 0.01    # 10mm 
s0 = 0.3
Emax = 0.002   # m/day
Zr = 0.1      # rooting depth, m

## rainfall simulations
gam = (n*Zr)/alpha
depths = simulate_rainfall(n_trajectories, tRun, dt, lam, gam) 
ps_samples, E_samples, L_samples = simulate_ps_t(n_trajectories, tRun, dt, s0, lam, alpha, Emax, Zr)

plt.figure(figsize=(5,6))
plt.subplot(411)
yesrain = depths[0]>0
plt.vlines(np.arange(len(depths[0]))[yesrain], 0,depths[0][yesrain]); plt.ylabel('Rainfall (mm)')
plt.subplot(412)
for v_t in ps_samples: 
    plt.plot(v_t, lw=1, color='lightgray')
plt.plot(ps_samples[0], lw=1, color='k'); plt.ylabel('Soil moisture')
plt.subplot(413)
for v_t in E_samples: 
    plt.plot(v_t, lw=1, color='lightgray')
plt.plot(E_samples[0], lw=1, color='k'); plt.ylabel('Evapotranspiration')
plt.subplot(414)
for v_t in L_samples: 
    plt.plot(v_t, lw=1, color='lightgray')
plt.plot(L_samples[0], lw=1, color='k'); plt.ylabel('Leakage')
plt.tight_layout()

nbins = 15
plt.figure(figsize=(3,6))
plt.subplot(311)
plt.hist(depths[depths>0].flatten(), normed=True, bins=nbins)
plt.subplot(312)
plt.hist(ps_samples.flatten(), normed=True, bins=nbins)
plt.subplot(313)
plt.hist(E_samples.flatten(), normed=True, bins=nbins)
plt.tight_layout()
plt.show()
plt.show()