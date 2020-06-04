
import pickle

# load MCMC output and thin
ts = pickle.load(open('../Data/UMB_trace/test.pickle', 'rb'))
for key in ts.keys():
    print('{}: {}'.format(key, ts[key].mean()))