
import numpy as np
import pandas as pd

# Read file
data = pd.read_csv('../Data/UMB_daily_average.csv', sep = ',')
data = data.replace(0, np.nan, regex=True)

# data availability
species = ['Aru', 'Bpa', 'Pgr', 'Pst', 'Qru']
columns = list(data.columns)#[col for col in list(data.columns) if(col[:3] in species)]
years = list(data['year'].drop_duplicates())
df = pd.DataFrame(columns=years, index=columns)
for yr in years:
    for col in columns:
        df.loc[col, yr] = data[col][data['year']==yr].count()
df = df.reset_index()
df = df.sort_values(by=['index'])
df.to_csv('../Data/UMB_data_selection.csv', index=False)
