
import numpy as np
import pandas as pd

# read file and preprocessing
year = 2015
df = pd.read_csv('../Data/raw_data/UMB_rawdata.csv')
df = df[df['year']==year]
df = df.iloc[9:-39, :]
df = df.replace(r'^\s+$', np.nan, regex=True)

# column names
ps_list = ['ps0', 'ps5', 'ps15', 'ps30', 'ps60', 'ps100', 'ps200', 'ps300']
swc_list = ['SWC_1_{}_1'.format(i+1) for i in range(8)]
species_dict = {'Aru_29': [123.64, 13.8], 'Bpa_38': [198.57, 18.1],
                'Pgr_22': [239.47, 21.6], 'Pst_19': [114.5, 15],
                'Qru_42': [460.3, 33.9]}
env_list = ['T', 'I', 'D']
date = ['year', 'month', 'day']

# the van Genuchten model
def psf(s, ss=0.37, sr=0.04, alpha=-5.2, n=1.68):
    return 1/alpha*(((ss-sr)/(s/100-sr))**(n/(n-1))-1)**(1/n)*0.009804139432
# day_len
def day_lenf(df, species):
    col = pd.DataFrame(df[species].values.reshape(-1, 48).T)
    col.iloc[:, list(col.isnull().all())] = \
    (col.iloc[:, list(col.isnull().all())]).fillna(0)
    sort = -(-col).transform(np.sort)
    sort = sort.cumsum()-sort.sum()*0.8
    res = sort[sort>=0].idxmin()+1
    res.name = '_'.join(['day_len', species])
    return res
# half hourly to daily
def maskf(df, species):
    col = pd.DataFrame(df[species].values.reshape(-1, 48).T)
    col.iloc[:, list(col.isnull().all())] = \
    (col.iloc[:, list(col.isnull().all())]).fillna(0)
    sort = -(-col).transform(np.sort)
    sort = sort.cumsum()-sort.sum()*0.8
    col.columns = sort[sort>=0].idxmin()+1
    col.columns = list(col.apply(lambda x: x.nlargest(x.name).iloc[-1]))
    mask = col.apply(lambda x: (x>=x.name)*1)
    return mask
def meanf(df, feature, species):
    mask = maskf(df, species)
    col = pd.DataFrame(df[feature].values.reshape(-1, 48).T)
    col = ((pd.DataFrame(mask.values*col.values)).replace(0, np.nan)).mean()
    col.name = '_'.join([feature, species])
    return col
def sumf(df, species):
    mask = maskf(df, species)
    col = pd.DataFrame(df[species].values.reshape(-1, 48).T)
    col = (pd.DataFrame(mask.values*col.values)).sum()
    col.name = species
    return col

# transformation
for ps, swc in zip(ps_list, swc_list):
    df[ps] = df.apply(lambda row: psf(row[swc]), axis=1)
df['D'] = df['D']/101.325
df_da = df[date].drop_duplicates()[:-1].reset_index()
df_da['date'] = pd.to_datetime((df_da['year']*10000+df_da['month']*100\
                               +df_da['day']).apply(str), format='%Y%m%d')
df_da = df_da.drop(['year', 'month', 'day', 'index'], axis=1)
feature_list = ps_list+env_list
for s in species_dict.keys():
    for f in feature_list:
        df_da = df_da.join(meanf(df, f, s))
    df_da = df_da.join(sumf(df, s))
    df_da = df_da.join(day_lenf(df, s))
    crown_area = 0.25*np.pi*(2.67*np.log(species_dict[s][1])-1.9)**2
    df_da[s] = df_da[s]*species_dict[s][0]/10000/crown_area*60*30/1000

# output
df_da.replace([np.inf, -np.inf], np.nan, inplace=True)
df_da.to_csv('../Data/UMB_daily_average_test_{}.csv'.format(year), index=False)

