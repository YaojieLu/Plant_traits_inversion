
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import PLCf

# read csv
df = pd.read_csv('../Data/UMB_daily_average_Gil_2015.csv')
# extract data
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', 'Aru_29']]
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)
df = df.iloc[54:86, ]
df = df.dropna()
T = df['T'].mean()
I = df['I'].mean()
D = df['D'].mean()
ps= df['ps'].mean()

def f1(px, c=30, p50=-5):
    PLC = PLCf(px, p50)
    f1 = lambda x:np.exp(-x*c)
    return (f1(PLC)-f1(1))/(f1(0)-f1(1))
def f2(px, c=.30, p50=-2):
    b=(0.3*p50-1)*(np.log(10))**(-1/c)
    return np.exp(-(px/b)**c)
def f3(px, p50=-1.2, c=10):
    return 2**(-(px/p50)**c)
x = np.linspace(-2, 0, 100)
y1 = f1(x)
y2 = f2(x)
plt.plot(x, y1, label='old')
plt.plot(x, y2, label='new')
plt.ylim(0, 1)
plt.xlabel('px', fontsize=30)
plt.ylabel('gs/gsmax', fontsize=30)
plt.legend(title='p50=-2; -0')
