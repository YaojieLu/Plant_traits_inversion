
import numpy as np
import pandas as pd

# read file and preprocessing
year = 2015
df = pd.read_csv('../Data/raw_data/UMB_rawdata.csv')
df = df[df['year']==year]
df = df.iloc[9:-39, :]
df = df.replace(r'^\s+$', np.nan, regex=True)

col = pd.DataFrame(df['Bpa_38'].values.reshape(-1, 48).T)
col.iloc[:, list(col.isnull().all())] = (col.iloc[:, list(col.isnull().all())]).fillna(0)
sort = -(-col).transform(np.sort)
sort = sort.cumsum()-sort.sum()*0.8
col.columns = sort[sort>=0].idxmin()+1
col.columns = list(col.apply(lambda x: x.nlargest(x.name).iloc[-1]))
mask = col.apply(lambda x: (x>=x.name)*1)
