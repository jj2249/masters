import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF
from matplotlib.colors import to_rgb
from scipy.stats import invgamma
from scipy.special import gamma

plt.style.use('ggplot')

### --- Forward Simulation --- ###

x0 = 0.
xd0 = 0.
mu0 = 0.
sigmasq = 1.
beta = 5.
kv = 5e-4
kmu = 1e-5
theta = -2.
p = 0.1
gsamps1 = 5_000

kw = 2.
rho = 1e-5
eta = 1e-5
N = 200
gsamps2 = 200
epsilon = 0.5

lss = LangevinModel(x0=x0, xd0=xd0, mu=mu0, sigmasq=sigmasq, beta=beta, kv=kv, kmu=kmu, theta=theta, p=p, gsamps=gsamps1)
lss.generate(nobservations=200)


## - store data in a dataframe - ##
sampled_dic = {'Telapsed': lss.observationtimes, 'Price': lss.observationvals}
sampled_data = pd.DataFrame(data=sampled_dic)

## - option to plot simulated data - ##

fig = plt.figure()
ax1 = fig.add_subplot(311)
ax1.plot(lss.observationtimes, lss.observationvals)
# ax1.set_xticks([])

ax2 = fig.add_subplot(312)
ax2.plot(lss.observationtimes, lss.observationgrad)
# ax2.set_xticks([])

ax3 = fig.add_subplot(313)
ax3.plot(lss.observationtimes, lss.observationmus)

ax1.set_ylabel(r'Position, $x$')
ax2.set_ylabel(r'Velocity, $\.x$')
ax3.set_ylabel(r'Skew, $\mu$')
plt.show()

### --- RBPF --- ###

## - define particle filter - ##

rbpf = RBPF(mux=x0, mumu=mu0, beta=beta, kw=kw, kv=kv, kmu=kmu, rho=rho, eta=eta, theta=theta, p=p, data=sampled_data, N=N, gsamps=gsamps2, epsilon=epsilon)
## - containers for storing results of rbpf - ##
fig = plt.figure()
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)
fig2 = plt.figure()
axxx = fig2.add_subplot(111)

## - main loop of rbpf - ##
sm, sv, gm, gv, mm, mv, lml, ax, mode, mean = rbpf.run_filter(ret_history=True, plot_marginal=True, ax=axxx, tpred=1.)

T = 0

## - plotting results of rbpf - ##
ax1.plot(rbpf.times[T:-1], lss.observationvals[T:], label='true')
ax1.plot(rbpf.times[T:], sm[T:], label='inferred')
# ax1.fill_between(rbpf.times[T:], (sm-1.96*np.sqrt(1.*sv))[T:], (sm+1.96*np.sqrt(1.*sv))[T:], color='orange', alpha=0.3)

ax2.plot(rbpf.times[T:-1], lss.observationgrad[T:])
ax2.plot(rbpf.times[T:], gm[T:])
# ax2.fill_between(rbpf.times[T:], (gm-1.96*np.sqrt(gv))[T:], (gm+1.96*np.sqrt(gv))[T:], color='orange', alpha=0.3)

ax3.plot(rbpf.times[T:-1], lss.observationmus[T:])
ax3.plot(rbpf.times[T:], mm[T:])
# ax3.fill_between(rbpf.times[T:], (mm-1.96*np.sqrt(mv))[T:], (mm+1.96*np.sqrt(mv))[T:], color='orange', alpha=0.3)

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


