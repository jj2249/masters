import numpy as np
from src.particlefilter import RBPF
from tqdm import tqdm
import sys


class PMMH:
	def __init__(self, mux, mumu, kw, kv, rho, eta, data, N, epsilon, delta, sampleX=False):
		# initial parameter vector -- beta, theta, kv
		# self.phi = np.array([10., -10.])
		# self.phi = np.array([10.*np.random.rand()+0.01, -10.*np.random.rand()-0.01])
		## lphi = {log (beta), log (-theta)}
		self.lphi = np.log(np.array([10.*np.random.rand()+1e-3, 10.*np.random.rand()+1e-3]))
		# RBPF parameters
		self.mux = mux
		self.mumu = mumu
		self.kw = kw
		self.kv = kv
		self.kmu = 0.
		self.rho = rho
		self.eta = eta
		self.p = 0.
		self.data = data
		self.N = N
		self.epsilon = epsilon

		self.delta = delta

		rbpf = RBPF(self.mux, self.mumu, np.exp(self.lphi[0]), self.kw, self.kv, self.kmu, self.rho, self.eta, -1.*np.exp(self.lphi[1]), self.p, self.data, self.N, self.epsilon)
		
		if sampleX:
			self.X, self.lml = rbpf.run_filter(sample=True)
		else:
			self.lml = rbpf.run_filter()

		# scaling for the Gaussian Random walk
		self.GRW = np.linalg.cholesky(delta*np.eye(2))
		# self.GRW = np.linalg.cholesky(np.array([[0.001, 0.],[0., 0.001]]))

		self.lphis = [self.lphi]
		if sampleX:
			self.Xs = [self.X]
		self.sampleX = sampleX

	def run_sampler(self, nits, tuning_interval=100.):
		accs = 0
		cnt = 0
		temp_accs = 0
		steps_before_tune = tuning_interval
		for _ in tqdm(range(nits)):
			lphistar = self.lphi - ((self.delta**2)/2.) + self.GRW @ np.random.randn(2) ## make lognormal
			if lphistar[0] > -6.89: ## constrains beta to be greater than ~1e-3 by immediate rejection
				rbpf = RBPF(self.mux, self.mumu, np.exp(lphistar[0]), self.kw, self.kv, self.kmu, self.rho, self.eta, -1.*np.exp(lphistar[1]), self.p, self.data, self.N, self.epsilon)
				if self.sampleX:
					Xstar, lmlstar = rbpf.run_filter(sample=True)
				else:
					lmlstar = rbpf.run_filter()

				a = np.minimum(0., lmlstar-self.lml + 2.*np.sum(lphistar) - 2.*np.sum(self.lphi)) # make lognormal
				val = np.log(np.random.rand())

				if a > val:
					if self.sampleX:
						self.X = Xstar
					self.lml = lmlstar
					self.lphi = lphistar
					accs += 1
					temp_accs += 1
			cnt += 1
			steps_before_tune -= 1
			if self.sampleX:
				self.Xs.append(self.X)
			self.lphis.append(self.lphi)
			if steps_before_tune == 0:
				acc_rate = (temp_accs)/(tuning_interval-steps_before_tune)
				print("\rInterval acceptance rate: " + str(acc_rate))
				self.tune_rate(acc_rate)
				steps_before_tune = tuning_interval
				temp_accs = 0
			print("\rOverall acceptance Rate: " + str(accs/cnt))
			print(np.array([np.exp(self.lphi[0]), -1.*np.exp(self.lphi[1])]))
		if self.sampleX:
			return self.Xs, self.lphis
		else:
			return (accs/cnt, np.array(self.lphis))


	def tune_rate(self, acc_rate):
		old_delta = self.delta
		if acc_rate < 0.001:
			self.delta *= 0.1
		elif acc_rate < 0.05:
			self.delta *= 0.5
		elif acc_rate < 0.2:
			self.delta *= 0.9
		elif acc_rate > 0.95:
			self.delta *= 10.
		elif acc_rate > 0.75:
			self.delta *= 2.0
		elif acc_rate > 0.5:
			self.delta *= 1.1

		print("Tuning step size: " + str(round(old_delta, 4)) + " --> " + str(round(self.delta, 4)))