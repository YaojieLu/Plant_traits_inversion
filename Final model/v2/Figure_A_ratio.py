import pandas as pd
import os
import seaborn as sns
# half violin plot; ridge plot
species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]

df = None
for species_code in species_codes:
    if not os.path.exists('../../Results/ensemble_A_UMB_Gil_v2_{}.csv'\
                          .format(species_code)):
        continue
    temp = pd.read_csv(open('../../Results/ensemble_A_UMB_Gil_v2_{}.csv'\
                          .format(species_code), 'rb'))
    temp['species_code'] = species_code
    temp['species'] = species_code[:3]
    if df is None:
        df = temp
    else:
        df = pd.concat([df, temp])
df = df.sort_values(by=['species_code'])
#df = df[df['species_code'] != 'Bpa_31']

# figure
sns.boxplot(x='species_code', y='qt=0.5', hue='species', data=df, width=3)
print(df.groupby('species_code')['qt=0.5'].mean())