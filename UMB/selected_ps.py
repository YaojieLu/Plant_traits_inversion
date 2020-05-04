
import pandas as pd

year = 2015
species_list = ['Aru', 'Bpa', 'Pgr', 'Pst', 'Qru']
species_dict = {'Aru':'Aru_29', 'Bpa':'Bpa_38', 'Pgr':'Pgr_27', 'Pst':'Pst_2', 'Qru':'Qru_10'}

# read csv
df = pd.read_csv('../Data/UMB_daily_average.csv')
df = df[df['year']==year]

# select ps based on correlation coefficient
ps_list = [col for col in df.columns if ('ps' in col) and (col != 'ps0') and (df[col].count()>0.9*len(df))]
df_corr = df.corr()
ps_selected = []

for sp in species_list:
    species_code = species_dict.get(sp)
    corr = dict(zip(ps_list, [df_corr.loc[species_code, ps] for ps in ps_list]))
    ps_selected.append(max(corr, key=corr.get))

print(dict(zip(species_list, ps_selected)))