
import numpy as np
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt

# read csv file
df = pd.read_csv('../Data/raw_data/UMB_rawdata.csv')
df['datetime']=pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute']])
df['time'] = (df.datetime-df.datetime[0]).dt.total_seconds()
# the van Genuchten model
def psf(s, ss=0.37, sr=0.04, alpha=-5.2, n=1.68):
    return 1/alpha*(((ss-sr)/(s/100-sr))**(n/(n-1))-1)**(1/n)*0.009804139432
df['ps'] = df[['SWC_1_3_1']].apply(psf)
env = 'I'
df = df[[env, 'time', 'datetime']]
df = df.dropna()
sig = df[env]

# compute and detect power
# The FFT of the signal
sig_fft = fftpack.fft(sig.values)
# And the power (sig_fft is of complex dtype)
power = np.abs(sig_fft)
# The corresponding frequencies
time_step = 0.5
sample_freq = fftpack.fftfreq(sig.size, d=time_step)
# Plot the FFT power
pos_mask = np.where(sample_freq>0)
plt.figure(figsize=(6, 5))
plt.plot(sample_freq[pos_mask], power[pos_mask])
plt.xlabel('Frequency [Hz]')
plt.ylabel('plower')

# remove all the high frequency
high_freq_fft = sig_fft.copy()
high_freq_fft[np.abs(sample_freq)>0.1] = 0
filtered_sig = np.real(fftpack.ifft(high_freq_fft))

fig, axs = plt.subplots(1, 2, figsize=(20, 5))
axs[0].plot(df['datetime'], sig, label='Original signal')
axs[0].plot(df['datetime'], filtered_sig, linewidth=3, label='Filtered signal')
axs[0].set_xlabel('Days')
axs[0].set_ylabel('Amplitude')
axs[0].legend(loc='best')
axs[1].scatter(sig, filtered_sig)
axs[1].set_xlabel('Original signal')
axs[1].set_ylabel('Filtered signal')
