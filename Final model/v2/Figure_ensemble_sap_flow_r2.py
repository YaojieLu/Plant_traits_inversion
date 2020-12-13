import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
for species_code in species_codes:
    if not os.path.exists('../../Results/ensemble_vn_UMB_Gil_v2_{}.csv'\
                          .format(species_code)):
        species_codes.remove(species_code)
species_codes.sort()

r2 = []
for species_code in species_codes:
        df = pd.read_csv('../../Results/ensemble_vn_UMB_Gil_v2_{}.csv'\
                         .format(species_code))
        df['observed'] = df['observed'].replace({0: np.nan})
        r2.append(df['qt=0.5'].corr(df['observed']))

r2 = pd.DataFrame({'species_code': species_codes, 'r2': r2})
r2['species'] = r2['species_code'].apply(lambda x: x[:3])