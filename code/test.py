import numpy as np
from numpy.matlib import repmat
import os
from tqdm import tqdm
import pandas as pd
from src.datahandler import TimeseriesData
from src.process import *
from src.particlefilter import RBPF
import matplotlib as mpl
# mpl.use("pgf")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
plt.style.use('seaborn')
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False
})
from scipy.stats import invgamma
from scipy.special import gamma


from copy import copy

tw = 6.50127
th = 9.00177
### --- Forward Simulation --- ###

x0 = 0.
xd0 = 0.
mu0 = 0.
sigmasq = 1.
beta = 1.
kv = 1e-4
kmu = 1e-15
theta = -3.
p = 0.
# gsamps1 = 100_000

kw = 1.
rho = 1e-5
eta = 1e-5
N = 500
# gsamps2 = 500
epsilon = 0.5

nobs = 100

lss = LangevinModel(x0=x0, xd0=xd0, mu=mu0, sigmasq=sigmasq, beta=beta, kv=kv, kmu=kmu, theta=theta, p=p)
lss.generate(nobservations=nobs)


## - store data in a dataframe - ##
sampled_dic = {'DateTime': lss.observationtimes, 'Bid': lss.observationvals}
sampled_data = pd.DataFrame(data=sampled_dic)

## - option to plot simulated data - ##
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax1.plot(lss.observationtimes, lss.observationvals)
# ax1.set_xticks([])
# ax1.xaxis.set_tick_params(length=0)
plt.setp(ax1.get_xticklabels(), visible=False)

ax2 = fig.add_subplot(212)
ax2.plot(lss.observationtimes, lss.observationgrad)
# ax2.set_xticks([])

ax1.set_ylabel(r'Position, $x$')
ax2.set_ylabel(r'Velocity, $\dot{x}$')
ax2.set_xlabel(r'Time, $t$')
# plt.show()
fig.set_size_inches(w=tw, h=0.5*tw)
plt.tight_layout()
# plt.savefig('../resources/figures/bmsde.pgf')
plt.show()
### --- RBPF --- ###

## - define particle filter - ##

rbpf = RBPF(mux=x0, mumu=0., beta=beta, kw=kw, kv=kv, kmu=kmu, rho=rho, eta=eta, theta=theta, p=p, data=sampled_data, N=N, epsilon=epsilon)
## - containers for storing results of rbpf - ##
fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)
# ax3 = fig.add_subplot(313)
# plt.tight_layout()
fig2 = plt.figure()
axxx = fig2.add_subplot(111)
fig3 = plt.figure()
axxx3 = fig3.add_subplot(111)
## - main loop of rbpf - ##
sm, sv, gm, gv, mm, mv, lml, ax, mode, mean, dss, pss, MSEs = rbpf.run_filter(ret_history=True, plot_marginal=True, ax=axxx, tpred=0.1, progbar=True, smin=0.1, smax=5.)
T = 0

## - plotting results of rbpf - ##
# ax1.plot(rbpf.times[T:-1], lss.observationvals[T:], label='true')
# ax1.plot(rbpf.times[T:], sm[T:], label='inferred')
# ax1.fill_between(rbpf.times[T:], (sm-1.96*np.sqrt(mode*sv))[T:], (sm+1.96*np.sqrt(mode*sv))[T:], color='orange', alpha=0.3)
# plt.setp(ax1.get_xticklabels(), visible=False)

# ax2.plot(rbpf.times[T:-1], lss.observationgrad[T:])
# ax2.plot(rbpf.times[T:], gm[T:])
# ax2.fill_between(rbpf.times[T:], (gm-1.96*np.sqrt(mode*gv))[T:], (gm+1.96*np.sqrt(mode*gv))[T:], color='orange', alpha=0.3)
# plt.setp(ax2.get_xticklabels(), visible=False)

# ax3.plot(rbpf.times[T:-1], lss.observationmus[T:])
# ax3.plot(rbpf.times[T:], mm[T:])
# ax3.fill_between(rbpf.times[T:], (mm-1.96*np.sqrt(mode*mv))[T:], (mm+1.96*np.sqrt(mode*mv))[T:], color='orange', alpha=0.3)

ax1.set_ylabel(r'Position, $x$')
ax2.set_ylabel(r'Velocity, $\dot{x}$')
# ax3.set_ylabel(r'Skew, $\mu$')
ax2.set_xlabel(r'Time, $t$')


## - full history - ##
# states, grads, skews, lweights = rbpf.run_filter_full_hist()
# ax1.plot(rbpf.times, lss.observationvals, lw=1.5)
# ax1.plot(rbpf.times, states, alpha=0.05, ls='-')
# ax2.plot(rbpf.times, lss.observationgrad, lw=1.5, label='True')
# ax2.plot(rbpf.times, grads, alpha=0.05, ls='-')
# ax3.plot(rbpf.times, lss.observationmus, lw=1.5)
# ax3.plot(rbpf.times, skews, alpha=0.05, ls='-')
# ax1.set_xticks([])
# ax2.set_xticks([])
# fig.legend()
# ax1.plot(rbpf.times, np.sum(states*np.exp(lweights), axis=1))
# ax2.plot(rbpf.times, np.sum(grads*np.exp(lweights), axis=1), label='Mixture Mean')
# ax3.plot(rbpf.times, np.sum(skews*np.exp(lweights), axis=1))
# smooth out each particle path
# times = rbpf.times.to_numpy()
# times = np.array(rbpf.times)
# num_fine = 2000
# t_fine = np.linspace(np.min(times), np.max(times), num_fine)
# ax1.axvline(x=times[-2], linestyle='--', color='orange')
# ax2.axvline(x=times[-2], linestyle='--', color='orange')
# states_fine = np.empty((num_fine, N), dtype=float)
# grads_fine = np.empty((num_fine, N), dtype=float)
# skews_fine = np.empty((num_fine, N), dtype=float)
# for i in range(N):
# 	states_fine[:,i] = np.interp(t_fine, times, states[:,i])
# 	grads_fine[:,i] = np.interp(t_fine, times, grads[:,i])
# 	skews_fine[:,i] = np.interp(t_fine, times, skews[:,i])
# states_fine = (states_fine.T).flatten()
# grads_fine = (grads_fine.T).flatten()
# skews_fine = (skews_fine.T).flatten()
# t_fine = repmat(t_fine, N, 1).flatten()
# cmap = copy(plt.cm.plasma)
# cmap.set_bad(cmap(0))


# sh, sxedges, syedges = np.histogram2d(t_fine, states_fine, bins=[400, 200], density=True)
# pcm = ax1.pcolormesh(sxedges, syedges, sh.T, cmap=cmap, norm=LogNorm(vmax=1.5), rasterized=True)
# gh, gxedges, gyedges = np.histogram2d(t_fine, grads_fine, bins=[400, 200], density=True)
# pcm = ax2.pcolormesh(gxedges, gyedges, gh.T, cmap=cmap, norm=LogNorm(vmax=1.5), rasterized=True)
# skh, skxedges, skyedges = np.histogram2d(t_fine, skews_fine, bins=[400, 200], density=True)
# pcm = ax3.pcolormesh(skxedges, skyedges, skh.T, cmap=cmap, norm=LogNorm(vmax=1.5), rasterized=True)
# fig.colorbar(pcm, ax=ax1)
# fig.legend(loc='upper left')

axxx.set_xlabel(r'Scale, $\sigma^2$')
axxx.set_ylabel(r'log $p(\sigma^2 | y_{1:N})$')
fig.suptitle(r'State Vector Filtering: $p(\alpha_t | y_{1:t}, \sigma^2)$')
# fig2.suptitle(r'Posterior Scale Density, mode, mean: ' +str(round(mode, 2)) + ', ' +str(round(mean, 2)))
print(mean, mode)
axxx3.plot(np.exp(dss), label=r'$D_N^{(\infty)}$')
axxx3.plot(np.exp(pss), label=r'$P_N^{(2)}$')
axxx3.axhline(y=epsilon*N, linestyle='--', color='black')
fig3.legend()
# fig.set_size_inches(w=tw, h=0.9*2.*th/3.)
fig2.set_size_inches(w=0.5*tw, h=th/3.)
fig3.set_size_inches(w=0.5*tw, h=th/3.)

plt.tight_layout()

# plt.savefig('../resources/figures/skewedallparticles.pgf')
plt.show()