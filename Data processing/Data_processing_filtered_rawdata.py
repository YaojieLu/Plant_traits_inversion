
import numpy as np
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt

# read csv file
df = pd.read_csv('../Data/raw_data/UMB_rawdata.csv')
df = df.replace(r'^\s*$', np.nan, regex=True)
df['D'] = df['D']/100

# fft
def fft_filter(sig, time_step, filtered_freq):
    # compute and detect power
    sig_fft = fftpack.fft(sig.values)
    sample_freq = fftpack.fftfreq(sig.size, d=time_step)
    # remove all the high frequency
    high_freq_fft = sig_fft.copy()
    high_freq_fft[np.abs(sample_freq)>filtered_freq] = 0
    filtered_sig = np.real(fftpack.ifft(high_freq_fft))
    return filtered_sig
time_step = 0.5
envs = ['T', 'I', 'D']#+['SWC_1_{}_1'.format(i+1) for i in range(8)]
for env in envs:
    df2 = df[[env]].copy()
    df2 = df2.dropna()
    sig = df2[env]
    df2['{}_filtered'.format(env)] = fft_filter(sig, time_step, 0.1)
    df = pd.concat([df, df2[['{}_filtered'.format(env)]]], axis=1, sort=False)

# the van Genuchten model
def psf(s, ss=0.37, sr=0.04, alpha=-5.2, n=1.68):
    return 1/alpha*(((ss-sr)/(s/100-sr))**(n/(n-1))-1)**(1/n)*0.009804139432
df_ps = df[['SWC_1_{}_1'.format(i+1) for i in range(8)]].apply(psf)
df_ps.columns = ['ps_{}'.format(i+1) for i in range(8)]
df = pd.concat([df, df_ps], axis=1, sort=False)
df['ps'] = df[['ps_4']].mean(axis=1)

# daily average
def daf(x):
    xre = x.values.reshape(-1, 48)
    da = np.nanmean(xre[:, 24:33], axis=1)
    return da

df_da = df[['year', 'month', 'day', 'T_filtered', 'I_filtered', 'D_filtered', 'ps', 'Aru_29']].apply(daf)
#df_da.replace([np.inf, -np.inf], np.nan, inplace=True)
df_da = df_da.dropna()
df_da.to_csv('../Data/UMB_daily_average_filtered.csv', sep = ',', index=False)
