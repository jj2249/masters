import numpy as np
import matplotlib.pyplot as plt
from datahandler import TimeseriesData
from particlefilter import RBPF

from p_tqdm import p_umap
from functools import partial
from itertools import product
import os


def filt_grid_search(theta, beta, data):
	return RBPF(mux=35000., mumu=0., beta=beta, kw=2., kv=10., kmu=1e-2, rho=1e-5, eta=1e-5, theta=theta, data=data, N=100, gsamps=200, epsilon=0.5).run_filter_MP()


### need to make sure that process spawning only happens once
if __name__ == '__main__':
	plt.style.use('ggplot')
	### --- importing data --- ###

	## - import data from a .csv - ##
	tobj = TimeseriesData(os.pardir+"/resources/data/oildata2.csv")
	
	# lss = LangevinModel(x0=0., xd0=0., mu=0., sigmasq=1., beta=0.8, kv=1e-6, kmu=1e-6, theta=-15., gsamps=100)
	# lss.generate(nobservations=200)
	Gt = 10
	Gb = 15

	thetas = np.linspace(-6., -2, Gt)
	betas = np.linspace(15., 60., Gb)
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
	
	fig = plt.figure()
	ax = fig.add_subplot()
	ax.contourf(theta_vals, beta_vals, lml_vals)
	# ax.plot(theta_vals, lml_vals)
	# ax.set_xlabel('theta')
	# ax.set_ylabel('lml')
	plt.show()

### --- RBPF --- ###


## - define particle filter - ##

# rbpf = RBPF(mux=110., mumu=0., beta=5., kw=2., kv=1., kmu=1e-2, rho=1e-5, eta=1e-5, theta=-5., data=tobj.df, N=500, gsamps=100, epsilon=0.5)

# # ## - containers for storing results of rbpf - ##
# fig = plt.figure()
# ax1 = fig.add_subplot(311)
# ax2 = fig.add_subplot(312)
# ax3 = fig.add_subplot(313)

# # ## - main loop of rbpf - ##
# sm, sv, gm, gv, mm, mv, lml = rbpf.run_filter(ret_history=True, tpred=15.)

# T = 0

# ### - plotting results of rbpf - ##
# ax1.plot(tobj.df['Telapsed'][T:], tobj.df['Price'][T:], label='true')
# ax1.plot(rbpf.times[T:], sm[T:])
# ax1.fill_between(rbpf.times[T:], (sm-1.96*np.sqrt(sv))[T:], (sm+1.96*np.sqrt(sv))[T:], color='orange', alpha=0.3)
# ax2.plot(rbpf.times[T:], gm[T:])
# ax2.fill_between(rbpf.times[T:], (gm-1.96*np.sqrt(gv))[T:], (gm+1.96*np.sqrt(gv))[T:], color='orange', alpha=0.3)
# ax3.plot(rbpf.times[T:], mm[T:])
# ax3.fill_between(rbpf.times[T:], (mm-1.96*np.sqrt(mv))[T:], (mm+1.96*np.sqrt(mv))[T:], color='orange', alpha=0.3)


# ax1.set_xticks([])
# ax2.set_xticks([])
# fig.legend()
# plt.show()
