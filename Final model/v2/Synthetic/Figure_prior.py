import pickle
from scipy.stats import truncnorm
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

# load MCMC output
baseline = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                            .format('baseline'), 'rb'))
p50 = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                       .format('prior_p50'), 'rb'))['p50']
c = pickle.load(open('../../../Data/UMB_trace/synthetic/{}.pickle'\
                     .format('prior_c'), 'rb'))['c']
x_p50 = np.linspace(-5, -0.1, 1000)
x_c = np.linspace(5, 30, 1000)
def get_truncated_normal(mean=-2.55, sd=1, low=-5, upp=-0.1):
    return truncnorm(
        (low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)
rv_p50 = get_truncated_normal()
rv_c = get_truncated_normal(mean=17.5, sd=5, low=5, upp=30)
true_p50 = -2.5
true_c = 13

# figure
fig, (ax1, ax2) = plt.subplots(2, 2, figsize=(20, 20))
# p50
ax1[0].hist(baseline['p50'], bins=30, color='cornflowerblue', density=True)
ax1[0].plot([-5, -0.1], [1/4.9, 1/4.9], color='black')
ax1[0].axvline(x=true_p50, linewidth=4, color='brown')
ax1[0].tick_params(axis='x', which='major', labelsize=20)
ax1[0].set_ylabel('Uniform prior', labelpad=30, fontsize=30)
ax1[0].axes.get_xaxis().set_ticks([])
ax1[0].axes.get_yaxis().set_ticks([])
ax2[0].hist(p50, bins=30, color='cornflowerblue', density=True)
ax2[0].plot(x_p50, rv_p50.pdf(x_p50), color='black')
ax2[0].axvline(x=true_p50, linewidth=4, color='brown')
ax2[0].tick_params(axis='x', which='major', labelsize=20)
ax2[0].set_xlabel('P50 (MPa)', fontsize=30)
ax2[0].axes.get_yaxis().set_ticks([])
ax2[0].set_ylabel('Truncated normal prior', labelpad=30, fontsize=30)
# c
ax1[1].hist(baseline['c'], bins=30, color='cornflowerblue', density=True)
ax1[1].plot([5, 30], [1/25, 1/25], color='black')
ax1[1].axvline(x=true_c, linewidth=4, color='brown')
ax1[1].tick_params(axis='x', which='major', labelsize=20)
ax1[1].axes.get_xaxis().set_ticks([])
ax1[1].axes.get_yaxis().set_ticks([])
ax2[1].hist(c, bins=30, color='cornflowerblue', density=True)
ax2[1].plot(x_c, rv_c.pdf(x_c), color='black')
ax2[1].axvline(x=true_c, linewidth=4, color='brown')
ax2[1].tick_params(axis='x', which='major', labelsize=20)
ax2[1].set_xlabel(r'$c$', fontsize=30)
ax2[1].axes.get_yaxis().set_ticks([])
hist = mpatches.Patch(color='cornflowerblue', label='Posterior distribution')
custom_lines = [Line2D([0], [0], color='black', label='Prior distribution'),
                Line2D([0], [0], color='brown', label='True value', lw=4)]
fig.legend(loc='upper center', handles=[hist]+custom_lines,\
           prop={'size': 30}, ncol=3, bbox_to_anchor=(0.512, 0.95))
fig.subplots_adjust(hspace=0.03, wspace=0.03)
fig.savefig('../../../Figures/Prior.png', bbox_inches='tight')
