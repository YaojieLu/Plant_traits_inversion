import pickle
import pandas as pd

for species in ['baseline', 'small', 'large', 'prior']:
    ts = pickle.load(open('../../../Data/UMB_trace/{}.pickle'.format(species),\
                          'rb'))
    ts['par1'] = 10**ts['alpha_log10']/10**ts['g1_log10']
    
    df = pd.DataFrame.from_dict(ts)
    df = df[['par1', 'c_log10', 'kxmax_log10', 'p50']]
    print(species)
    print(df.mean())
    #print(df.var())
    #print(df.var()/df.mean())