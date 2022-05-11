import numpy as np
import matplotlib.pyplot as plt
from src.datahandler import TimeseriesData, TickData
from src.particlefilter import RBPF

from p_tqdm import p_umap
from functools import partial
from itertools import product
import os


plt.style.use('ggplot')
# tobj = TimeseriesData(os.pardir+"/resources/data/brentdata.csv", idx1=-160)
# plt.plot(tobj.df['Telapsed'], tobj.df['Price'])
# plt.show()

tobj = TickData(os.pardir+"/resources/data/EURGBP-2022-04.csv", nrows=250)
print(tobj.df)

### --- RBPF --- ###


# - define particle filter - ##
N = 1000
epsilon = 0.5
rbpf = RBPF(mux=0.84227, mumu=0., beta=4., kw=1e-7, kv=5e0, kmu=0., rho=1e-5, eta=1e-5, theta=-2., p=0., data=tobj.df, N=N, gsamps=200, epsilon=epsilon)

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
fig2 = plt.figure()
axxx = fig2.add_subplot(111)

## - main loop of rbpf - ##
sm, sv, gm, gv, mm, mv, lml, ax, mode, mean, dss, pss, MSEs = rbpf.run_filter(ret_history=True, plot_marginal=True, ax=axxx, tpred=0., progbar=True)

T = 0

## - plotting results of rbpf - ##
ax1.plot(rbpf.times[T:], rbpf.prices[T:], label='true')
ax1.plot(rbpf.times[T:], sm[T:], label='inferred')
ax1.fill_between(rbpf.times[T:], (sm-1.96*np.sqrt(mode*sv))[T:], (sm+1.96*np.sqrt(mode*sv))[T:], color='orange', alpha=0.3)

ax2.plot(rbpf.times[T:], gm[T:])
ax2.fill_between(rbpf.times[T:], (gm-1.96*np.sqrt(mode*gv))[T:], (gm+1.96*np.sqrt(mode*gv))[T:], color='orange', alpha=0.3)

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
fig3 = plt.figure()
axxx3 = fig3.add_subplot(111)
axxx3.plot(np.exp(dss), label=r'$D_N^{(\infty)}$')
axxx3.plot(np.exp(pss), label=r'$P_N^{(2)}$')
axxx3.axhline(y=epsilon*N, linestyle='--', color='black')
fig3.legend()
plt.show()