import numpy as np
from pmcmc import PMMH
from p_tqdm import p_umap
from multiprocessing import Process
from functools import partial
from process import LangevinModel
import pandas as pd

def sampler(container, mux, mumu, kw, rho, eta, data, N, gsamps, epsilon, delta, nits):
	pmmh = PMMH(mux, mumu, kw, rho, eta, data, N, gsamps, epsilon, delta, sampleX=False)
	phis = pmmh.run_sampler(nits)
	container.append(np.array(phis))

if __name__ == '__main__':
	### --- Forward Simulation --- ###

	x0 = 0.
	xd0 = 0.
	mu0 = 0.
	sigmasq = 1.
	beta = 5.
	kv = 5e-4
	kmu = 1e-5
	theta = -2.
	p = 0.
	gsamps1 = 5_000

	kw = 50.
	rho = 1e-5
	eta = 1e-5
	N = 200
	gsamps2 = 200
	epsilon = 0.5

	lss = LangevinModel(x0=x0, xd0=xd0, mu=mu0, sigmasq=sigmasq, beta=beta, kv=kv, kmu=kmu, theta=theta, p=p, gsamps=gsamps1)
	lss.generate(nobservations=50)


	## - store data in a dataframe - ##
	sampled_dic = {'Telapsed': lss.observationtimes, 'Price': lss.observationvals}
	sampled_data = pd.DataFrame(data=sampled_dic)

	results = []

	p = Process(target=sampler, args=(results, x0, mu0, kw, rho, eta, sampled_data, N, gsamps2, epsilon, 0.1, 5))
	p.start()
	p.join()
	# results = p_umap(partial(sampler, mux=x0, mumu=mu0, kw=kw, rho=rho, eta=eta, data=sampled_data, N=N, gsamps=gsamps2, epsilon=epsilon, delta=0.1, nits=5), [])
	np.save('./samples.npy', results)