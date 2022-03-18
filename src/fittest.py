import numpy as np
import matplotlib.pyplot as plt
from datahandler import TimeseriesData
from particlefilter import RBPF

from p_tqdm import p_umap
from functools import partial
from itertools import product
import os


plt.style.use('ggplot')
tobj = TimeseriesData(os.pardir+"/resources/data/brentdata.csv", idx1=-160)
# plt.plot(tobj.df['Telapsed'], tobj.df['Price'])
# plt.show()



### --- RBPF --- ###


# - define particle filter - ##

rbpf = RBPF(mux=125., mumu=0., beta=29., kw=1., kv=1e-1, kmu=0., rho=1e-5, eta=1e-5, theta=-4., p=0., data=tobj.df, N=500, gsamps=1000, epsilon=0.5)

## - containers for storing results of rbpf - ##
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
fig2 = plt.figure()
axxx = fig2.add_subplot(111)

## - main loop of rbpf - ##
sm, sv, gm, gv, mm, mv, lml, ax, mode, mean = rbpf.run_filter(ret_history=True, plot_marginal=True, ax=axxx, tpred=0.01)

T = 0

## - plotting results of rbpf - ##
ax1.plot(rbpf.times[T:-1], rbpf.prices[T:], label='true')
ax1.plot(rbpf.times[T:], sm[T:], label='inferred')
ax1.fill_between(rbpf.times[T:], (sm-1.96*np.sqrt(mode*sv))[T:], (sm+1.96*np.sqrt(mode*sv))[T:], color='orange', alpha=0.3)

ax2.plot(rbpf.times[T:], gm[T:])
# ax2.fill_between(rbpf.times[T:], (gm-1.96*np.sqrt(mode*gv))[T:], (gm+1.96*np.sqrt(mode*gv))[T:], color='orange', alpha=0.3)

ax3.plot(rbpf.times[T:], mm[T:])
ax3.fill_between(rbpf.times[T:], (mm-1.96*np.sqrt(mode*mv))[T:], (mm+1.96*np.sqrt(mode*mv))[T:], color='orange', alpha=0.3)

ax1.set_ylabel(r'Position, $x$')
ax2.set_ylabel(r'Velocity, $\.x$')
ax3.set_ylabel(r'Skew, $\mu$')
ax3.set_xlabel(r'Time, $t$')


## - full history - ##
# states, grads, skews = rbpf.run_filter_full_hist()
# ax1.plot(rbpf.times, lss.observationvals, lw=1.5)
# ax1.plot(rbpf.times, states, alpha=0.05, ls='-')
# ax2.plot(rbpf.times, lss.observationgrad, lw=1.5)
# ax2.plot(rbpf.times, grads, alpha=0.05, ls='-')
# ax3.plot(rbpf.times, lss.observationmus, lw=1.5)
# ax3.plot(rbpf.times, skews, alpha=0.05, ls='-')

# ax1.set_xticks([])
# ax2.set_xticks([])
fig.legend()


axxx.set_xlabel(r'Scale, $\sigma^2$')
axxx.set_ylabel(r'log $p(\sigma^2 | y_{1:N})$')
fig.suptitle(r'State Vector Filtering: $p(\alpha_t | y_{1:t}, \sigma^2)$')
fig2.suptitle(r'Posterior Scale Density, mode, mean: ' +str(mode) + ', ' +str(mean))
plt.show()