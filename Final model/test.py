import pandas as pd
import numpy as np

# read csv
species_code = 'Pgr_27'
df = pd.read_csv('../../Data/UMB_daily_average_Gil_v2.csv')
df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', 'day_len', species_code]]
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)
print(df.shape)
df[species_code] = df[species_code].replace({0: np.nan})
df = df.dropna()
print(df.shape)
df = df.drop(df.index[list(range(0, 10))])
print(df.shape)
