
import numpy as np
import pandas as pd

# Read file
df = pd.read_csv('../Data/raw_data/UMB_rawdata.csv')
df = df.replace(r'^\s*$', np.nan, regex=True)

# column names
ps_list = ['ps0', 'ps5', 'ps15', 'ps30', 'ps60', 'ps100', 'ps200', 'ps300']
swc_list = ['SWC_1_{}_1'.format(i+1) for i in range(8)]

# the van Genuchten model
def psf(s, ss=0.37, sr=0.04, alpha=-5.2, n=1.68):
    return 1/alpha*(((ss-sr)/(s/100-sr))**(n/(n-1))-1)**(1/n)*0.009804139432

# daily average
def daf(x):
    xre = x.values.reshape(-1, 48)
    ls = xre[:, 24:33]
    da = np.nanmean(ls, axis=1)
    return da

# preprocessing
for ps, swc in zip(ps_list, swc_list):
    df[ps] = df.apply(lambda row: psf(row[swc]), axis=1)
#df['ps'] = df[['ps3']].mean(axis=1)

# normalization
species = ['Aru', 'Bpa', 'Pgr', 'Pst', 'Qru']
columns = [col for col in list(df.columns) if(col[:3] in species)]
df[columns] = df[columns]/100
df['D'] = df['D']/100

# output
df = df.drop(['hour', 'minute'], axis=1)
df_da = df.apply(daf)
df_da.replace([np.inf, -np.inf], np.nan, inplace=True)
df_da.to_csv('../Data/UMB_daily_average.csv', index=False)
