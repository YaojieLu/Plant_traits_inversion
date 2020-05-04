
import pickle
import pandas as pd
from Simulation_model import UMBf
import matplotlib.pyplot as plt

# read trace
ts = pickle.load(open("../Data/UMB_trace/12.pickle", "rb"))

# read MCMC inputs
df = pd.read_csv('../Data/UMB_daily_average.csv')
T = df['T']
I = df['I']
D = df['D']
ps = df['ps']
vn1 = df['Aru_29']

# unique values
df = pd.DataFrame(data=ts)
df_unique = df.drop_duplicates()
df_unique['alpha'] = df_unique['alpha']*100
vn2 = UMBf(df_unique.iloc[1, :]*1.1, T, I, D, ps)
rss = ((vn1-vn2)**2).sum()
print(rss)

# sum of square error
rss = []
for i in range(len(df_unique)):
    vn2 = UMBf(df_unique.iloc[i, :], T, I, D, ps)
    rss.append(((vn1-vn2)**2).sum())

# figures
plt.plot(rss)