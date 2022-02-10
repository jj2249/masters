import numpy as np
import copy
from process import GammaProcess, LangevinModel
import pandas as pd


def logsumexp(x):
	"""
	Helper function for calculating the log of a sum of exponentiated values in a numerically stable way
	"""
	c = np.max(x)
	return c + np.log(np.sum(np.exp(x-c)))


class LangevinParticle(LangevinModel):
	"""
	Underlying particle object in the particle filter
	"""
	def __init__(self, mumu, sigmasq, beta, kw, kv, theta, gsamps):
		LangevinModel.__init__(self, mumu, sigmasq, beta, kv, theta, gsamps)
		# model parameters
		self.theta = theta
		self.kv = kv
		self.beta = beta
		self.sigmasq = sigmasq
		
		# implementation parameters
		self.gsamps = gsamps

		# initial kalman parameters
		# a current current
		# C current current
		self.acc = np.array([0, 0, mumu])
		self.Ccc = np.array([[0, 0, 0],[0, 0, 0],[0, 0, self.sigmasq*kw]])

		# sample initial state using cholesky decomposition
		Cc = np.linalg.cholesky(self.Ccc + 1e-12*np.eye(3))
		self.alpha = self.acc + Cc @ np.random.randn(3)

		# log particle weight
		self.logweight = 0.

		self.Hmat = self.H_matrix()
		self.Bmat = self.B_matrix()
		

	def __repr__(self):
		return str("acc: "+self.acc.__repr__()+'\n'
			+"Ccc: "+self.Ccc.__repr__()+'\n'
			+"Un-normalised weight: "+str(np.exp(self.logweight))
			)


	def increment(self, observation, s, t):
		if type(s) == pd._libs.tslibs.timedeltas.Timedelta:
				s = s.total_seconds()
		if type(t) == pd._libs.tslibs.timedeltas.Timedelta:
				t = t.total_seconds()

		# time interval between two observations
		dt = t - s

		# latent gamma process
		Z = GammaProcess(1., self.beta, samps=self.gsamps, minT=s, maxT=t)
		Z.generate()

		# parameters for estimating stochastic integral
		m = self.langevin_m(t, self.theta, Z)
		S = self.sigmasq*self.langevin_S(t, self.theta, Z)
		# come back to this if there are stability issues
		Sc = np.linalg.cholesky(S+1e-12*np.eye(2))
		e = Sc @ np.random.randn(2)

		Amat = self.A_matrix(m, dt)
		# print(Amat.shape)
		# state increment
		self.alpha = (Amat @ self.alpha) + (self.Bmat @ e)

		# prediction step
		acp = (Amat @ self.acc).reshape(-1, 1)
		Ccp = (Amat @ self.Ccc @ Amat.T) + (self.Bmat @ S @ self.Bmat.T)

		# Kalman gain
		K = (Ccp @ self.Hmat.T) / ((self.Hmat @ Ccp @ self.Hmat.T) + self.sigmasq*self.kv)
		K = K.reshape(-1, 1)

		# correction step
		self.acc = acp + (K * (observation - self.Hmat @ acp))
		self.Ccc = Ccp - (K @ self.Hmat @ Ccp)

		# Prediction Error Decomposition
		ayt = (self.Hmat @ acp)
		Cyt = (self.Hmat @ Ccp @ self.Hmat.T) + (self.sigmasq*self.kv)
		Cyt = Cyt.flatten()

		# update log weight
		# print(observation-ayt)
		self.logweight += -0.5 * np.log(2.*np.pi*Cyt) - (1./(2.*Cyt))*np.square(observation-ayt)



class RBPF:
	"""
	Full rao-blackwellised (marginalised) particle filter
	"""
	def __init__(self, mumu, sigmasq, beta, kw, kv, theta, data, N, gsamps, epsilon):

		# x and y values for the timeseries
		self.times = data['Date_Time']
		self.prices = data['Price']

		# store initial values
		self.initial_time = self.times[0]
		self.initial_price = self.prices[0]

		# transform data so that first observation is zero and t=0
		self.prices = self.prices.subtract(self.initial_price)
		self.times = self.times.subtract(self.initial_time)

		# generators for passing through the times and observations
		self.timegen = (time for time in self.times)
		self.pricegen = (price for price in self.prices)
		
		self.current_time = self.timegen.__next__()
		self.current_price = self.pricegen.__next__()

		# implementation parameters
		# no. of particles
		self.N = N
		# limit for resampling based on effective sample size
		self.log_resample_limit = np.log(self.N*epsilon)

		# collection of particles
		self.particles = [LangevinParticle(mumu, sigmasq, beta, kw, kv, theta, gsamps) for _ in range(N)]
		

	def reweight_particles(self):
		"""
		Renormalise particle weights to sum to 1
		"""
		lweights = np.array([particle.logweight for particle in self.particles])
		# numerically stable implementation
		sum_weights = logsumexp(lweights)
		for particle in self.particles:
			# log domain
			particle.logweight = particle.logweight - sum_weights


	def increment_particles(self):
		"""
		Increment each particle based on the newest time and observation
		"""
		# collect new times and prices
		self.current_price = self.pricegen.__next__()
		prev_time = self.current_time
		self.current_time = self.timegen.__next__()

		# reweight each particle -- could be faster using a map()?
		for particle in self.particles:
			particle.increment(self.current_price, prev_time, self.current_time)


	def resample_particles(self):
		"""
		Resample particles using multinomial distribution, then set weights to 1/N
		"""

		lweights = np.array([particle.logweight for particle in self.particles]).flatten()
		# normalised weights are the probabilities
		probabilites = np.nan_to_num(np.exp(lweights))

		# need to renormalise to account for any underflow when exponentiating -- better way to do this?
		probabilites = probabilites / np.sum(probabilites)
		
		# multinomial method returns an array with the number of selections stored at each location
		selections = np.random.multinomial(self.N, probabilites)
		new_particles = []
		# for each particle
		for idx in range(self.N):
			# copy this particle the appropriate amount of times
			for _ in range(selections[idx]):
				# print(idx)
				new_particles.append(copy.copy(self.particles[idx]))
		
		# overwrite old particles
		self.particles = new_particles
		
		# reset each weight
		for particle in self.particles:
			particle.logweight = -np.log(self.N)


	def get_state_mean(self):
		"""
		Get weighted sum of current particle means
		"""
		weights = np.array([np.exp(particle.logweight).reshape(1, -1) for particle in self.particles])
		means = np.array([particle.acc for particle in self.particles])
		return np.sum(weights*means, axis=0)


	def get_state_covariance(self):
		"""
		Get weighted sum of current particle variances
		"""
		weights = np.array([np.exp(particle.logweight).reshape(1, -1) for particle in self.particles])
		covs = np.array([particle.Ccc for particle in self.particles])
		return np.sum(weights*covs, axis=0)


	def get_logPn2(self):
		"""
		Inverse sum of squares for estimating ESS
		"""
		lweights = np.array([particle.logweight for particle in self.particles])
		return -np.log(np.sum(np.exp(2*lweights)))


	def get_logDninf(self):
		"""
		Inverse maximum weight for estimating ESS
		"""
		lweights = np.array([particle.logweight for particle in self.particles])
		return -np.max(lweights)

