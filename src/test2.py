import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF


### --- Demonstration of convergence of gamma process --- ###

# ## - running parameters - ##
# alpha = 1.
# beta = .01
# maxT = 5.

# ## - figures - ##
# fig = plt.figure()
# ax1 = fig.add_subplot(2,2,1)
# ax2 = fig.add_subplot(2,2,2)
# ax3 = fig.add_subplot(2,1,2)

# ## final_vals stores the end point of each generated process
# final_vals = []
# for i in tqdm(range(1000)):
# 	g = GammaProcess(1., beta, samps=5000, minT=0., maxT=5.)
# 	g.generate()
# 	g.plot_timeseries(ax1)
# 	final_vals.append(np.sum(g.jsizes))

# ## - axis - ##
# y = np.linspace(0., np.max(final_vals), 1000)
# ## - pdf - ##
# ax2.hist(final_vals, bins=50, density=True, cumulative=False, orientation='horizontal')
# g.marginal_gamma(y, 5., ax2)
# ## - cdf - ##
# ax3.hist(final_vals, bins=50, density=True, cumulative=True, orientation='vertical')
# g.marginal_gamma_cdf(y, 5., ax3)

# ## - plot titles - ###
# ax1.set_title('Skeleton Processes')
# ax1.set_xlabel('t')
# ax1.set_ylabel('Gt')
# ax2.set_title('Marginal PDF')
# ax2.set_ylabel('G1')
# ax2.set_xlabel('frequency')
# ax3.set_title('Marginal CDF')
# ax3.set_ylabel('frequency')
# ax3.set_xlabel('G1')
# fig.legend()
# plt.show()


### --- Test skeletons of the VG process --- ###

fig = plt.figure()
ax1 = fig.add_subplot(111)
ax1.set_xlabel('t')
ax1.set_ylabel('Zt')
for i in range(500):
	vg = VarianceGammaProcess(beta=1., mu=0, sigmasq=1.)
	vg.generate()
	vg.plot_timeseries(ax1)
plt.show()



### -- verification of VG process using Kernel-smoothed density -- ### RETURN TO THIS
# fig = plt.figure()
# ax2 = fig.add_subplot(111)
# mus = np.array([-5., -2.5, 0., 2.5, 5.])
# mus = np.array([-2., -1., 0., 1., 2.])
# mus = np.array([-3., 0., 3.])
# betas = np.array([0.1, 0.5, 1.])
# sigmas = np.array([0.5, 0.1, 1.])
# for mu in tqdm(mus):
# for beta in tqdm(betas):
# for sigma in tqdm(sigmas):
	# final_vals = []
	# for i in tqdm(range(10000)):
	# 	vg = VarianceGammaProcess(beta=beta, mu=0., sigmasq=1.)
	# 	vg.generate()
	# 	final_vals.append(np.sum(vg.jsizes))
	# y = np.linspace(np.min(final_vals), np.max(final_vals), 1000)
	# y = np.linspace(-6., 6., 10000)
	# ax2.hist(final_vals, density=True, bins=50, orientation='vertical')
	# ks_density = ksdensity(final_vals, width=0.3)
	# ax2.plot(y, ks_density(y))
	# vg.marginal_variancegamma(y, 1., ax2, label='mu: '+str(round(mu, 2)))
	# vg.marginal_variancegamma(y, 1., ax2, label='beta: '+str(round(beta, 2)))
	# vg.marginal_variancegamma(y, 1., ax2, label='sigmasq: '+str(round(sigma, 2)))


# ax2.set_xlabel('VG1')
# ax2.set_ylabel('frequency')
# fig.legend()
# plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# final_vals = []
# for i in range(1000):
# 	vg = VarianceGammaProcess(beta=.1, mu=0, sigmasq=1.)
# 	vg.generate()
# 	vg.plot_timeseries(ax1)
# 	final_vals.append(np.sum(vg.jsizes))
# y = np.linspace(np.min(final_vals), np.max(final_vals), 1000)
# ax2.hist(final_vals, bins=50, density=True, cumulative=False, orientation='horizontal')
# ks_density = ksdensity(final_vals, width=0.3)
# ax2.plot(ks_density(y), y)
# plt.show()
