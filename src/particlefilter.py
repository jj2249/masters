import numpy as np
import copy
from process import GammaProcess
import pandas as pd

def logsumexp(x):
	c = np.max(x)
	return c + np.log(np.sum(np.exp(x-c)))

class Particle:
	def __init__(self, P, mumu, sigmasq, beta, kw, kv, theta):
		self.theta = theta
		self.P = P
		self.kv = kv
		self.beta = beta

		# initial kalman parameters
		# a current current
		# C current current
		self.acc = np.zeros((P+1, 1))
		self.acc[-1] = mumu
		self.Ccc = np.zeros((P+1, P+1))
		self.Ccc[-1,-1] = sigmasq*kw
		self.sigmasq = sigmasq

		# initial state
		Cc = np.linalg.cholesky(self.Ccc + 1e-12*np.eye(self.P+1))
		self.alpha = self.acc + Cc @ np.random.randn(P+1)

		# log weight
		self.logweight = 0.

		self.Hmat = self.H_matrix()
		self.Bmat = self.B_matrix()
		self.lastobservation = 0.
		

	def __repr__(self):
		return str("acc: "+self.acc.__repr__()+'\n'
			+"Ccc: "+self.Ccc.__repr__()+'\n'
			+"Un-normalised weight: "+str(np.exp(self.logweight))
			)


	def A_matrix(self, Z, m, dt):
		return np.block([[Z.langevin_drift(dt, self.theta), m.reshape(-1, 1)],
						[np.zeros((1, self.P)), 1.]])


	def B_matrix(self):
		return np.vstack([np.eye(self.P),
						np.zeros((1, self.P))])


	def H_matrix(self):
		h = np.zeros((1, self.P+1))
		h[:, 0] = 1.
		return h



	def increment(self, observation, s, t):
		if type(s) == pd._libs.tslibs.timedeltas.Timedelta:
				s = s.total_seconds()
		if type(t) == pd._libs.tslibs.timedeltas.Timedelta:
				t = t.total_seconds()
		dt = t - s
		Z = GammaProcess(1., self.beta, samps=1000, minT=s, maxT=t)
		Z.generate()

		# m = self.lastobservation*Z.langevin_m(t, self.theta)
		# S = (self.lastobservation**2)*self.sigmasq*Z.langevin_S(t, self.theta)
		m = Z.langevin_m(t, self.theta)
		S = self.sigmasq*Z.langevin_S(t, self.theta)
		# come back to this if there a stability issues
		Sc = np.linalg.cholesky(S+1e-12*np.eye(self.P))
		e = Sc @ np.random.randn(self.P)

		Amat = self.A_matrix(Z, m, dt)

		self.alpha = (Amat @ self.alpha) + (self.Bmat @ e)
		# prediction step
		acp = Amat @ self.acc
		Ccp = (Amat @ self.Ccc @ Amat.T) + (self.Bmat @ S @ self.Bmat.T)
		# Kalman gain
		K = (Ccp @ self.Hmat.T) / ((self.Hmat @ Ccp @ self.Hmat.T) + self.sigmasq*self.kv)
		K = K.reshape(-1, 1)
		# correction step
		self.acc = acp + (K * (observation - self.Hmat @ acp))
		self.Ccc = Ccp - (K @ self.Hmat @ Ccp)
		# Prediction Error Decomposition
		ayt = self.Hmat @ acp
		Cyt = (self.Hmat @ Ccp @ self.Hmat.T) + (self.sigmasq*self.kv)
		Cyt = Cyt.flatten()
		self.logweight += -0.5 * np.log(2.*np.pi*Cyt) - (1./(2.*Cyt))*np.square(observation-ayt)

		self.lastobservation = self.alpha[0][0]


class RBPF:
	def __init__(self, P, mumu, sigmasq, beta, kw, kv, theta, data, N):

		self.times = data['Date_Time']
		self.prices = data['Price']

		self.initial_time = self.times[0]
		self.initial_price = self.prices[0]

		# set first observation to zero
		self.prices = self.prices.subtract(self.initial_price)
		self.times = self.times.subtract(self.initial_time)

		self.timegen = (time for time in self.times)
		self.pricegen = (price for price in self.prices)
		
		self.current_time = self.timegen.__next__()
		self.current_price = self.pricegen.__next__()

		self.N = N
		self.particles = [Particle(P, mumu, sigmasq, beta, kw, kv, theta) for _ in range(N)]
		

	def reweight_particles(self):
		lweights = np.array([particle.logweight for particle in self.particles])
		# come back to this if there are underflow problems
		sum_weights = logsumexp(lweights)
		for particle in self.particles:
			# log domain
			particle.logweight = particle.logweight - sum_weights


	def increment_particles(self):
		self.current_price = self.pricegen.__next__()
		prev_time = self.current_time
		self.current_time = self.timegen.__next__()
		# print("Prev time: "+str(prev_time)
		# 	+'\n'+"Current time: "+str(self.current_time)
		# 	+'\n'+"Current price: "+str(self.current_price))
		for particle in self.particles:
			particle.increment(self.current_price, prev_time, self.current_time)


	def resample_particles(self):
		lweights = np.array([particle.logweight for particle in self.particles]).flatten()
		probabilites = np.nan_to_num(np.exp(lweights))

		probabilites = probabilites / np.sum(probabilites)
		# print(probabilites)
		selections = np.random.multinomial(self.N, probabilites)
		new_particles = []
		for idx in range(self.N):
			for _ in range(selections[idx]):
				new_particles.append(copy.copy(self.particles[idx]))
		self.particles = new_particles
		for particle in self.particles:
			particle.logweight = -np.log(self.N)


	def get_state_mean(self):
		weights = np.array([np.exp(particle.logweight).reshape(1, -1) for particle in self.particles])
		means = np.array([particle.acc for particle in self.particles])
		return np.sum(weights*means, axis=0)


	def get_all_means(self):
		means = [particle.acc for particle in self.particles]
		return means[:][0]


	def get_state_covariance(self):
		weights = np.array([np.exp(particle.logweight).reshape(1, -1) for particle in self.particles])
		covs = np.array([particle.Ccc for particle in self.particles])
		return np.sum(weights*covs, axis=0)