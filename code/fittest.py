import numpy as np
from numpy.matlib import repmat
from src.datahandler import TimeseriesData, TickData
from src.particlefilter import RBPF
import matplotlib as mpl
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False
})
import matplotlib.pyplot as plt
from p_tqdm import p_umap
from functools import partial
from itertools import product
import os
from copy import copy
from matplotlib.colors import LogNorm


plt.style.use('ggplot')
# tobj = TimeseriesData(os.pardir+"/resources/data/brentdata.csv", idx1=-160)
# plt.plot(tobj.df['Telapsed'], tobj.df['Price'])
# plt.show()

tobj = TickData(os.pardir+"/resources/data/CHFJPY-2022-04.csv", nrows=308)
print(tobj.df)
fig = plt.figure()
ax = fig.add_subplot()
tobj.plot(ax)
fig.show()

### --- RBPF --- ###

tw = 6.50127
th = 9.00177
# - define particle filter - ##
N = 500
epsilon = 0.5
rbpf = RBPF(mux=131.865, mumu=0., beta=1.003, kw=1e-10, kv=5., kmu=0., rho=1e-5, eta=1e-5, theta=-0.2778, p=0., data=tobj.df, N=N, epsilon=epsilon)

fig = plt.figure()
ax1 = fig.add_subplot(111)
fig3 = plt.figure()
ax2 = fig3.add_subplot(111)
# ax3 = fig.add_subplot(313)
fig2 = plt.figure()
axxx = fig2.add_subplot(111)

## - main loop of rbpf - ##
sm, sv, gm, gv, mm, mv, lml, ax, mode, mean, dss, pss, MSEs = rbpf.run_filter(ret_history=True, plot_marginal=True, ax=axxx, tpred=0., progbar=True)

T = 0

## - plotting results of rbpf - ##
ax1.plot(rbpf.times[T:], rbpf.prices[T:], label='true', ls='--', lw=0., marker='.', ms=2., mec='black', mfc='black')
ax1.plot(rbpf.times[T:], sm[T:], label='inferred')
ax1.fill_between(rbpf.times[T:], (sm-1.96*np.sqrt(mode*sv))[T:], (sm+1.96*np.sqrt(mode*sv))[T:], color='orange', alpha=0.3)

ax2.plot(rbpf.times[T:], gm[T:])
ax2.fill_between(rbpf.times[T:], (gm-1.96*np.sqrt(mode*gv))[T:], (gm+1.96*np.sqrt(mode*gv))[T:], color='orange', alpha=0.3)

# ax3.plot(rbpf.times[T:], mm[T:])
# ax3.fill_between(rbpf.times[T:], (mm-1.96*np.sqrt(mode*mv))[T:], (mm+1.96*np.sqrt(mode*mv))[T:], color='orange', alpha=0.3)

ax1.set_ylabel(r'Position, $x$')
ax2.set_ylabel(r'Velocity, $\dot{x}$')
# ax3.set_ylabel(r'Skew, $\mu$')
ax1.set_xlabel(r'Time, $t$')
ax2.set_xlabel(r'Time, $t$')

# states, grads, skews, lweights = rbpf.run_filter_full_hist(progbar=True)
# ax1.plot(rbpf.times[T:], rbpf.prices[T:], color='black', ls='--', lw=0., marker='.', ms=2., mec='black', mfc='black')
# ax1.set_xticks([])
# ax2.set_xticks([])
# ax1.plot(rbpf.times[T:], np.sum(states*np.exp(lweights), axis=1)[T:], color='red', ls='-.')
# ax2.plot(rbpf.times[T:], np.sum(grads*np.exp(lweights), axis=1)[T:], label='Mixture Mean', color='red', ls='-.')
# ax3.plot(rbpf.times[T:], np.sum(skews*np.exp(lweights), axis=1)[T:], color='red', ls='-.')
# smooth out each particle path
# times = rbpf.times.to_numpy()
# times = np.array(rbpf.times)
# num_fine = 2000
# t_fine = np.linspace(np.min(times), np.max(times), num_fine)
# # ax1.axvline(x=times[-2], linestyle='--', color='orange')
# # ax2.axvline(x=times[-2], linestyle='--', color='orange')
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
# cmap = copy(plt.cm.Greys)
# cmap.set_bad(cmap(0))


# sh, sxedges, syedges = np.histogram2d(t_fine, states_fine, bins=[400, 200], density=True)
# pcm = ax1.pcolormesh(sxedges, syedges, sh.T, cmap=cmap, norm=LogNorm(vmax=1.5), rasterized=True)
# gh, gxedges, gyedges = np.histogram2d(t_fine, grads_fine, bins=[400, 200], density=True)
# pcm = ax2.pcolormesh(gxedges, gyedges, gh.T, cmap=cmap, norm=LogNorm(vmax=1.5), rasterized=True)
# skh, skxedges, skyedges = np.histogram2d(t_fine, skews_fine, bins=[400, 200], density=True)
# pcm = ax3.pcolormesh(skxedges, skyedges, skh.T, cmap=cmap, norm=LogNorm(vmax=1.5), rasterized=True)

# axxx.set_xlabel(r'Scale, $\sigma^2$')
# axxx.set_ylabel(r'log $p(\sigma^2 | y_{1:N})$')
# fig.suptitle(r'State Vector Filtering: $p(\alpha_t | y_{1:t}, \sigma^2)$')
# fig2.suptitle(r'Posterior Scale Density, mode, mean: ' +str(mode) + ', ' +str(mean))
# fig3 = plt.figure()
# axxx3 = fig3.add_subplot(111)
# axxx3.plot(np.exp(dss), label=r'$D_N^{(\infty)}$')
# axxx3.plot(np.exp(pss), label=r'$P_N^{(2)}$')
# axxx3.axhline(y=epsilon*N, linestyle='--', color='black')
# fig3.legend()
ax1.get_yaxis().get_major_formatter().set_useOffset(False)
ax1.get_yaxis().get_major_formatter().set_scientific(False)
ax1.get_xaxis().get_major_formatter().set_useOffset(False)
ax1.get_xaxis().get_major_formatter().set_scientific(False)
fig.set_size_inches(h=0.8*0.5*tw, w=1.1*0.9*th/2.)
fig3.set_size_inches(h=0.8*0.5*tw, w=1.1*0.9*th/2.)
fig.tight_layout()
fig3.tight_layout()
plt.show()