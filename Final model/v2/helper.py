import pickle
import matplotlib.pyplot as plt
import pandas as pd

# helper function
def f(name):
    ts = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                          .format(name), 'rb'))
    ts['kxmax/g1'] = 10**ts['kxmax_log10']/10**ts['g1_log10']
    ts['kxmax/alpha'] = 10**ts['kxmax_log10']/10**ts['alpha_log10']
    df = pd.DataFrame.from_dict(data=ts, orient='columns')
    df = df.drop(['g1_log10', 'kxmax_log10', 'alpha_log10'], axis=1)
    return df

df = f('large')
print(df.mean())