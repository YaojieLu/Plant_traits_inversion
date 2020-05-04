
import numpy as np
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt

# read csv file
df = pd.read_csv('../Data/raw_data/UMB_rawdata.csv')

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
envs = ['T', 'I', 'D']
for env in envs:
    df2 = df[[env]].copy()
    df2 = df2.dropna()
    sig = df2[env]
    df2['{}_filtered'.format(env)] = fft_filter(sig, time_step, 0.1)
    df = pd.concat([df, df2[['{}_filtered'.format(env)]]], axis=1, sort=False)

# figure
fig, axs = plt.subplots(1, 3, figsize=(30, 5))
for i in range(3):
    df.plot.scatter(x=envs[i], y='{}_filtered'.format(envs[i]), legend=False, ax=axs[i])
df.to_csv('../Data/raw_data/UMB_rawdata_filtered.csv', sep = ',', index=False)
