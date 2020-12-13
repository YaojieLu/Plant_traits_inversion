import numpy as np
import pandas as pd
from scipy import optimize
from Functions import kxf, pxminf, pxf2
import matplotlib.pyplot as plt
import seaborn as sns

# read MCMC input
df = pd.read_csv('../../../Data/UMB_daily_average_Gil_v2.csv')
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', 'day_len']]
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)

# model
def muf(c, p50, T, I, D, ps, day_len,
        g1_log10=1, kxmax_log10=3,
        Vcmax=60, Jmax=120, ca=400, Kc=460, q=0.3, R=8.314, z1=0.9, z2=0.9999,\
        a=1.6, L=1):
    g1, kxmax = 10**g1_log10, 10**kxmax_log10
    pxmin = pxminf(ps, p50)
    pxmax = optimize.minimize_scalar(pxf2, bounds=(pxmin, ps),\
                                     method='bounded',\
                                     args=(T, I, D, ps, Kc, Vcmax, ca, q,\
                                           Jmax, z1, z2, R, g1, c, kxmax, p50,\
                                           a, L)).x
    px1 = pxf2(pxmin, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c,\
               kxmax, p50, a, L)
    px2 = pxf2(pxmax, T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1, z2, R, g1, c,\
               kxmax, p50, a, L)
    if px1*px2 < 0:
        px = optimize.brentq(pxf2, pxmin, pxmax,\
                             args=(T, I, D, ps, Kc, Vcmax, ca, q, Jmax, z1,\
                                   z2, R, g1, c, kxmax, p50, a, L))
        return kxf(px, kxmax, p50)*(ps-px)*day_len
    return np.nan

# simulation
param_c = np.linspace(5, 30, 51)
param_p50 = np.linspace(-10, -0.5, 39)

def f():
    sen = []
    for idx, row in df.iterrows():
        if idx%10 == 0:
            print(idx)
        T, I, D, ps, day_len = row['T'], row['I'], row['D'], row['ps'],\
                               row['day_len']
        temp = [muf(c, p50, T, I, D, ps, day_len) for c in param_c\
                for p50 in param_p50]
        sen.append(temp)
    return sen
sen=f()
df=pd.DataFrame([[c, p50] for c in param_c for p50 in param_p50],
                columns=['c', 'p50'])
df['avg_vn'] = pd.DataFrame(sen).T.mean(axis=1)
df_test = df#[df['p50']<-1]
df_test = df_test.pivot(index='c', columns='p50', values='avg_vn')

# figure
sns.set(font_scale=1.3)
fig = plt.figure(figsize=(16, 16))
ax = sns.heatmap(df_test, cmap='viridis', xticklabels=10, yticklabels=10,\
                  cbar_kws={'label': '$\\mathit{v_{n}}$'})
plt.xlabel('$\\psi_{x50}$', fontsize=30)
plt.ylabel('$\\mathit{c}$', fontsize=30)
ax.figure.axes[-1].yaxis.label.set_size(30)
plt.savefig('../../../Figures/Figure_Sensitivity_c_p50_Gil_v2.png',\
            bbox_inches='tight')
