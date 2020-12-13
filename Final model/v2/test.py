import numpy as np
import pandas as pd

species_code = 'Bpa_41'

df = pd.read_csv('../../Data/UMB_daily_average_Gil_v2.csv')

df = df[['T', 'I', 'D', 'ps15', 'ps30', 'ps60', species_code]]
df['ps'] = df[['ps15', 'ps30', 'ps60']].mean(1)
df[species_code] = df[species_code].replace({0: np.nan})
df = df.dropna()
df = df.drop(df.index[[61, 62, 69, 76, 77, 78, 79, 81, 82]])