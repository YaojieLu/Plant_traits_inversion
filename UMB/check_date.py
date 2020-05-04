
import pandas as pd

# species
species, year = 'Pgr', 2015
species_dict = {'Aru':'Aru_29', 'Bpa':'Bpa_38', 'Pgr':'Pgr_27', 'Pst':'Pst_2', 'Qru':'Qru_10'}
species_code = species_dict.get(species)

# read csv
df = pd.read_csv('../Data/UMB_daily_average.csv')
df = df[df['year']==year]

# selected ps
ps_selected = 'ps5'

# extract data
df = df[['T', 'I', 'D', ps_selected, species_code]]
df = df.dropna()
df = df.drop(df.index[[49, 51]])
for i in range(len(df)):
    print(i, df.iloc[i, 0])