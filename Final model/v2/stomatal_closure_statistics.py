import pandas as pd
import os

species_codes = {1: 'Pst', 2: 'Pst', 3: 'Pgr', 4: 'Pgr', 6: 'Pst',
                 7: 'Aru', 9: 'Aru', 11: 'Aru', 12: 'Pst', 14: 'Pst',
                 17: 'Pst', 19: 'Pst', 21: 'Aru', 22: 'Pgr', 24: 'Aru',
                 26: 'Aru', 27: 'Pgr', 28: 'Aru', 29: 'Aru', 31: 'Bpa',
                 32: 'Bpa', 34: 'Pgr', 35: 'Bpa', 38: 'Bpa', 41: 'Bpa'}
species_codes = [y+'_'+str(x) for x, y in species_codes.items()]
for species_code in species_codes:
    if not os.path.exists('../../Results/ensemble_gs_PLC_UMB_Gil_v2_2_{}.csv'\
                          .format(species_code)):
        species_codes.remove(species_code)
species_codes.sort()

# figure
species_abbrs = ['Aru', 'Bpa', 'Pgr', 'Pst']
species_names = ['Red maple', 'Paper birch', 'Bigtooth aspen',\
                 'Eastern white pine']
species_names = dict(zip(species_abbrs, species_names))
i = 0
for species_code in species_codes:
    df = pd.read_csv('../../Results/ensemble_gs_PLC_UMB_Gil_v2_2_{}.csv'\
                     .format(species_code))
    df = df.dropna()
    df['gs'] = df['gs'].apply(lambda x: 0.05*round(x/0.0005))
    df_5 = df.groupby('gs')['PLC'].quantile(.05)
    df_50 = df.groupby('gs')['PLC'].median()
    df_95 = df.groupby('gs')['PLC'].quantile(.95)
    df = pd.concat([df_5, df_50, df_95], axis=1, join='inner')
    df.reset_index(inplace=True)
    df.columns = ['gs', '5', '50', '95']
    #print(df)
    print(df[df['gs']==10][['5', '50', '95']])