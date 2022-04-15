import numpy as np
from particlefilter import RBPF
from tqdm import tqdm
import sys


class PMMH:
	def __init__(self, mux, mumu, kw, rho, eta, data, N, gsamps, epsilon, delta, sampleX=False):
		# initial parameter vector -- beta, theta, kv
		self.phi = np.array([.5, -.5, 0.5])

		# RBPF parameters
		self.mux = mux
		self.mumu = mumu
		self.kw = kw
		self.kmu = 0.
		self.rho = rho
		self.eta = eta
		self.p = 0.
		self.data = data
		self.N = N
		self.gsamps = gsamps
		self.epsilon = epsilon

		rbpf = RBPF(self.mux, self.mumu, self.phi[0], self.kw, self.phi[2], self.kmu, self.rho, self.eta, self.phi[1], self.p, self.data, self.N, self.gsamps, self.epsilon)
		
		if sampleX:
			self.X, self.lml = rbpf.run_filter(sample=True)
		else:
			self.lml = rbpf.run_filter()

		# scaling for the Gaussian Random walk
		# self.GRW = np.linalg.cholesky(delta*np.eye(3))
		self.GRW = np.linalg.cholesky(np.array([[0.001, 0., 0.],[0., 0.001, 0.],[0., 0., 0.001]]))

		self.phis = [self.phi]
		if sampleX:
			self.Xs = [self.X]
		self.sampleX = sampleX

	def run_sampler(self, nits):
		accs = 0.
		cnt = 0.
		for _ in tqdm(range(nits)):
			phistar = self.phi + self.GRW @ np.random.randn(3)
			rbpf = RBPF(self.mux, self.mumu, phistar[0], self.kw, phistar[2], self.kmu, self.rho, self.eta, phistar[1], self.p, self.data, self.N, self.gsamps, self.epsilon)
			if self.sampleX:
				Xstar, lmlstar = rbpf.run_filter(sample=True)
			else:
				lmlstar = rbpf.run_filter()

			a = np.minimum(0., lmlstar-self.lml)
			val = np.log(np.random.rand())

			if a > val:
				if self.sampleX:
					self.X = Xstar
				self.lml = lmlstar
				self.phi = phistar
				accs += 1.
			cnt += 1.
			if self.sampleX:
				self.Xs.append(self.X)
			self.phis.append(self.phi)
			print("\rAcceptance Rate: " + str(accs/cnt))
			print(self.phi)
		if self.sampleX:
			return self.Xs, self.phis
		else:
			return self.phis