import numpy as np
import matplotlib.pyplot as plt
from src.datahandler import TimeseriesData, TickData
from src.particlefilter import RBPF

from p_tqdm import p_umap
from functools import partial
from itertools import product
import os


def filt_grid_search(theta, beta, data):
	return RBPF(mux=0.84227, mumu=0., beta=beta, kw=1e-10, kv=10., kmu=0., rho=1e-5, eta=1e-5, theta=theta, p=0., data=data, N=200, epsilon=0.5).run_filter_MP()


### need to make sure that process spawning only happens once
if __name__ == '__main__':
	plt.style.use('ggplot')
	### --- importing data --- ###

	## - import data from a .csv - ##
	tobj = TickData(os.pardir+"/resources/data/EURGBP-2022-04.csv", nrows=200)
	
	# lss = LangevinModel(x0=0., xd0=0., mu=0., sigmasq=1., beta=0.8, kv=1e-6, kmu=1e-6, theta=-15., gsamps=100)
	# lss.generate(nobservations=200)
	Gt = 25
	Gb = 25

	thetas = np.linspace(-25., -0.01, Gt)
	betas = np.linspace(.02, 4., Gb)
	grid = np.array(list(product(thetas, betas)))
	theta_vals = grid[:,0]
	beta_vals = grid[:,1]
	
	results = p_umap(partial(filt_grid_search, data=tobj.df), theta_vals, beta_vals)
	
	results = np.array(results)
	theta_vals = results[:,0]
	beta_vals = results[:,1]
	lml_vals = results[:,2]

	idx = np.argsort(theta_vals, axis=0)
	theta_vals = np.take(theta_vals, idx)
	beta_vals = np.take(beta_vals,idx)
	lml_vals = np.take(lml_vals, idx)

	theta_vals = theta_vals.reshape(Gb, Gt, order='F')
	beta_vals = beta_vals.reshape(Gb, Gt, order='F')
	lml_vals = lml_vals.reshape(Gb, Gt, order='F')

	idx2 = np.argsort(beta_vals, axis=0)
	theta_vals = np.take_along_axis(theta_vals, idx2, axis=0)
	beta_vals = np.take_along_axis(beta_vals, idx2, axis=0)
	lml_vals = np.take_along_axis(lml_vals, idx2, axis=0)
	maxidx = np.unravel_index(lml_vals.argmax(), lml_vals.shape)
	print("Beta: "+str(beta_vals[maxidx]))
	print("Theta: "+str(theta_vals[maxidx]))

	fig = plt.figure()
	ax = fig.add_subplot()
	ax.contourf(theta_vals, beta_vals, lml_vals)
	# ax.plot(theta_vals, lml_vals)
	# ax.set_xlabel('theta')
	# ax.set_ylabel('lml')
	plt.show()