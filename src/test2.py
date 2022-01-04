import numpy as np
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import pandas as pd
from datahandler import TimeseriesData
from process import *
from particlefilter import RBPF
from ksdensity import ksdensity

mu = 1.
nu = .1
# maxT = 1.
# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# # ax = fig.add_subplot()
# ax2 = fig.add_subplot(122)
# final_vals = []

# for i in range(1000):
# 	g = GammaProcess(mu**2/nu, mu/nu, samps=1000, minT=0., maxT=1.)
# 	g.generate()
# 	g.plot_timeseries(ax1)
# 	final_vals.append(np.sum(g.jsizes))

# y = np.linspace(0., np.max(final_vals), 1000)
# ax2.hist(final_vals, bins=50, density=True, cumulative=True, orientation='horizontal')
# g.marginal_gamma_cdf(y, 1., ax2)

# fig.legend()
# plt.show()

# fig = plt.figure()
# ax1 = fig.add_subplot(121)
# ax2 = fig.add_subplot(122)
# final_vals = []
# for i in range(1000):
# 	vg = VarianceGammaProcess(C=1./nu, beta=1./nu, theta=1., sigmasq=1.)
# 	vg.generate()
# 	vg.plot_timeseries(ax1)
# 	final_vals.append(np.sum(vg.jsizes))
# y = np.linspace(np.min(final_vals), np.max(final_vals), 1000)
# ax2.hist(final_vals, bins=50, density=True, cumulative=False, orientation='horizontal')
# ks_density = ksdensity(final_vals, width=0.3)
# ax2.plot(ks_density(y), y)
# plt.show()


# theta_vals = np.linspace(-2., 2., 5)
# fig = plt.figure()
# ax = fig.add_subplot()
# for theta in tqdm(theta_vals):
# 	final_vals = []
# 	for i in tqdm(range(1000)):
# 		vg = VarianceGammaProcess(C=1./nu, beta=1./nu, theta=theta, sigmasq=1.)
# 		vg.generate()
# 		final_vals.append(np.sum(vg.jsizes))
# 	y = np.linspace(np.min(final_vals), np.max(final_vals), 1000)
# 	ks_density = ksdensity(final_vals, width=0.3)
# 	ax.plot(ks_density(y), label=str(round(theta, 2)))
# fig.legend()
# plt.show()

nus = np.linspace(1., 5., 5)
fig = plt.figure()
ax = fig.add_subplot()
for nu in tqdm(nus):
	final_vals = []
	for i in tqdm(range(10000)):
		vg = VarianceGammaProcess(C=1./nu, beta=1./nu, theta=0., sigmasq=1.)
		vg.generate()
		final_vals.append(np.sum(vg.jsizes))
	y = np.linspace(np.min(final_vals), np.max(final_vals), 1000)
	ks_density = ksdensity(final_vals, width=0.3)
	ax.plot(ks_density(y), label=str(round(nu, 2)))
fig.legend()
plt.show()