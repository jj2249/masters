import numpy as np
from src.pmcmc import PMMH
from p_tqdm import p_umap
from functools import partial
from src.process import LangevinModel
import pandas as pd
import matplotlib.pyplot as plt


def sampler(mux, mumu, kw, kv, rho, eta, data, N, epsilon, delta, nits):
	pmmh = PMMH(mux, mumu, kw, kv, rho, eta, data, N, epsilon, delta, sampleX=False)
	phis = pmmh.run_sampler(nits)
	return phis

def throwaway(num):
	return sampler()

if __name__ == '__main__':
	### --- Forward Simulation --- ###

	x0 = 0.
	xd0 = 0.
	mu0 = 0.
	sigmasq = 1.
	beta = 0.8
	kv = 1e-3
	kmu = 0.
	theta = -2.
	p = 0.

	kw = 1.
	rho = 1e-5
	eta = 1e-5
	N = 500
	epsilon = 0.5

	lss = LangevinModel(x0=x0, xd0=xd0, mu=mu0, sigmasq=sigmasq, beta=beta, kv=kv, kmu=kmu, theta=theta, p=p)
	lss.generate(nobservations=100)

	fig = plt.figure()
	ax1 = fig.add_subplot(211)
	ax1.plot(lss.observationtimes, lss.observationvals)
	
	plt.setp(ax1.get_xticklabels(), visible=False)

	ax2 = fig.add_subplot(212)
	ax2.plot(lss.observationtimes, lss.observationgrad)
	ax1.set_xticks([])

	ax1.set_ylabel(r'Position, $x$')
	ax2.set_ylabel(r'Velocity, $\dot{x}$')
	ax2.set_xlabel(r'Time, $t$')
	fig.set_size_inches(w=6., h=0.5*6.)
	plt.tight_layout()
	plt.show()


	## - store data in a dataframe - ##
	sampled_dic = {'DateTime': lss.observationtimes, 'Bid': lss.observationvals}
	sampled_data = pd.DataFrame(data=sampled_dic)

	results = p_umap(partial(sampler, mumu=mu0, kw=kw, kv=kv, rho=rho, eta=eta, data=sampled_data, N=N, epsilon=epsilon, delta=.45, nits=3000), x0*np.ones(8))
	np.savez('./samples6.npz', results, lss, pickle=True)