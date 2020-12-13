import pickle
from sklearn.linear_model import LinearRegression
import pandas as pd
from matplotlib import pyplot as plt

# load MCMC output and thin
species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]

df = []
for species_code in species_codes:
    try:
        ts = pickle.load(open('../../Data/UMB_trace/Gil_v2/{}.pickle'\
                              .format(species_code), 'rb'))
    except FileNotFoundError:
        continue
    X = (10**ts['alpha_log10']).reshape(-1, 1)
    y= 10**ts['kxmax_log10']
    reg = LinearRegression().fit(X, y)
    df.append([species_code, reg.score(X, y), reg.coef_[0]])

df = pd.DataFrame(df, columns=['species_code', 'r2', 'slope'])
ax = df.plot.bar(x='species_code', y='r2', rot=0)
plt.xticks(fontsize=15, rotation=45)
plt.yticks(fontsize=15)