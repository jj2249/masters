import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import pandas as pd

import os
from p_tqdm import p_umap
from functools import partial
from itertools import product

from process import *
from particlefilter import RBPF
from datahandler import TimeseriesData


def info(title):
    print(title)
    print('module name:', __name__)
    print('parent process:', os.getppid())
    print('process id:', os.getpid())



def filt_grid_search(theta, beta, data):
	return RBPF(mux=0., mumu=0., beta=beta, kw=1., kv=1e-4, kmu=1e-6, rho=1., eta=1., theta=theta, p=0., data=data, N=200, gsamps=200, epsilon=0.5).run_filter_MP()


def filt_grid_search_kv(kv, data):
	return RBPF(mux=0., mumu=0., beta=2., kw=1., kv=kv, kmu=1e-6, rho=1., eta=1., theta=-2., p=0., data=data, N=200, gsamps=200, epsilon=0.5).run_filter_kv()


### need to make sure that process spawning only happens once
if __name__ == '__main__':
	info('main line')
	
	lss = LangevinModel(x0=0., xd0=0., mu=0., sigmasq=1., beta=2., kv=1e-4, kmu=1e-6, theta=-2., p=0., gsamps=1000)
	lss.generate(nobservations=200)

	G = 200

	## - import data from a .csv - ##

	# data = TimeseriesData(os.pardir+"/resources/data/test_data.csv")
	# df_u = data.remove_non_unique(ret=True)
	# plt.plot(df_u['Date_Time'], df_u['Price'])
	# plt.xticks([])
	# plt.show()

	# thetas = np.linspace(-25., -.1, G)
	# betas = np.linspace(0.05, 2.1, G)
	# grid = np.array(list(product(thetas, betas)))
	# theta_vals = grid[:,0]
	# beta_vals = grid[:,1]
	# ## - store data in a dataframe - ##
	
	# sampled_dic = {'Date_Time': lss.observationtimes, 'Price': lss.observationvals}
	# sampled_data = pd.DataFrame(data=sampled_dic)
	
	# ## - option to plot simulated data - ##
	
	fig = plt.figure()
	ax1 = fig.add_subplot(311)
	ax2 = fig.add_subplot(312)
	ax3 = fig.add_subplot(313)
	ax1.plot(lss.observationtimes, lss.observationvals)
	ax2.plot(lss.observationtimes, lss.observationgrad)
	ax3.plot(lss.observationtimes, lss.observationmus)
	ax1.set_xticks([])
	plt.show()

	# results = p_umap(partial(filt_grid_search, data=df_u), theta_vals, beta_vals)
	
	# results = np.array(results)
	# theta_vals = results[:,0]
	# beta_vals = results[:,1]
	# lml_vals = results[:,2]

	# idx = np.argsort(theta_vals, axis=0)
	# theta_vals = np.take(theta_vals, idx)
	# beta_vals = np.take(beta_vals,idx)
	# lml_vals = np.take(lml_vals, idx)

	# theta_vals = theta_vals.reshape(G, G, order='F')
	# beta_vals = beta_vals.reshape(G, G, order='F')
	# lml_vals = lml_vals.reshape(G, G, order='F')

	# idx2 = np.argsort(beta_vals, axis=0)
	# theta_vals = np.take_along_axis(theta_vals, idx2, axis=0)
	# beta_vals = np.take_along_axis(beta_vals, idx2, axis=0)
	# lml_vals = np.take_along_axis(lml_vals, idx2, axis=0)
	
	# fig = plt.figure()
	# ax = fig.add_subplot()
	# ax.contourf(theta_vals, beta_vals, lml_vals)
	# # ax.plot(theta_vals, lml_vals)
	# # ax.set_xlabel('theta')
	# # ax.set_ylabel('lml')
	# plt.show()

	kvs = np.logspace(-6., 2., G)
	print(kvs)

	sampled_dic = {'Telapsed': lss.observationtimes, 'Price': lss.observationvals}
	sampled_data = pd.DataFrame(data=sampled_dic)

	results = p_umap(partial(filt_grid_search_kv, data=sampled_data), kvs)
	results = np.array(results)

	kv_vals = results[:,0]
	lml_vals = results[:,1]
	idx = np.argsort(kv_vals, axis=0)
	kv_vals = np.take(kv_vals, idx)
	lml_vals = np.take(lml_vals, idx)

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.semilogx(kv_vals, lml_vals)
	plt.show()
