import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# load MCMC output
vn_0_ps_0 = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                             .format('vn_0_ps_0'), 'rb'))['p50']
vn_0_ps_05 = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                             .format('vn_0_ps_05'), 'rb'))['p50']
vn_05_ps_0 = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                             .format('vn_05_ps_0'), 'rb'))['p50']
vn_05_ps_05 = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                             .format('vn_05_ps_05'), 'rb'))['p50']
truc_p50 = -2.5

# figure
colors = ['#3300cc', '#ff0000']
fig, axs = plt.subplots(1, 2, figsize=(20, 10), sharey=True)
axs[0].hist(vn_0_ps_0, bins=30, color=colors[0],\
            histtype='step', density=True)
axs[0].hist(vn_0_ps_05, bins=30, color=colors[1],\
            histtype='step', density=True)
axs[0].axvline(x=truc_p50, linewidth=4, color='brown')
axs[0].tick_params(axis='x', which='major', labelsize=20)
#axs[0].set_xlim([5, 30])
axs[0].set_xlabel(r'$c$', fontsize=30)
axs[0].yaxis.set_visible(False)
axs[0].set_title('Uniform prior', fontsize=30)
axs[1].hist(prior, bins=30, color='cornflowerblue', density=True)
axs[1].plot(x, rv.pdf(x), color='black')
axs[1].axvline(x=truc_c, linewidth=4, color='brown')
axs[1].tick_params(axis='x', which='major', labelsize=20)
axs[1].set_xlabel(r'$c$', fontsize=30)
axs[1].yaxis.set_visible(False)
axs[1].set_title('Truncated normal prior', fontsize=30)
hist = mpatches.Patch(color='cornflowerblue', label='Posterior distribution')
custom_lines = [Line2D([0], [0], color='black', label='Prior distribution'),
                Line2D([0], [0], color='brown',\
                       label=r'True value of $c$: {}'.format(truc_c), lw=4)]
axs[0].legend(handles=[hist]+custom_lines, prop={'size': 20})
fig.subplots_adjust(wspace=0.03)
fig.savefig('../../../Figures/Prior.png', bbox_inches='tight')
