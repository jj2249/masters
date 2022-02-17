import numpy as np
import copy
from process import GammaProcess, LangevinModel
import pandas as pd
from tqdm import tqdm


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
	def __init__(self, mumu, beta, kw, kv, theta, gsamps):
		LangevinModel.__init__(self, mumu, 1., beta, kv, theta, gsamps)
		# model parameters
		self.theta = theta
		self.kv = kv
		self.beta = beta
		
		# implementation parameters
		self.gsamps = gsamps

		# initial kalman parameters
		# a current current
		# C current current
		self.acc = np.array([0, 0, mumu])
		# self.Ccc = np.array([[0, 0, 0],[0, 0, 0],[0, 0, self.sigmasq*kw]])
		self.Ccc = np.array([[0, 0, 0],[0, 0, 0],[0, 0, kw]])


		# sample initial state using cholesky decomposition
		Cc = np.linalg.cholesky(self.Ccc + 1e-12*np.eye(3))
		# self.alpha = self.acc + Cc @ np.random.randn(3)

		# log particle weight
		self.logweight = 0.

		self.Hmat = self.H_matrix()
		self.Bmat = self.B_matrix()


	def __repr__(self):
		return str(#"alpha: "+self.alpha.__repr__()+'\n'
			"acc: "+self.acc.__repr__()+'\n'
			+"Ccc: "+self.Ccc.__repr__()+'\n'
			+"Un-normalised weight: "+str(np.exp(self.logweight))
			)


	def predict(self, s, t):
		# time interval between two observations
		dt = t - s

		# latent gamma process
		Z = GammaProcess(1., self.beta, samps=self.gsamps, minT=s, maxT=t)
		Z.generate()

		# parameters for estimating stochastic integral
		m = self.langevin_m(t, self.theta, Z)
		# S = self.sigmasq*self.langevin_S(t, self.theta, Z)
		S = self.langevin_S(t, self.theta, Z)

		Amat = self.A_matrix(m, dt)
		# print(Amat)
		# prediction step
		self.acp = (Amat @ self.acc).reshape(-1, 1)
		self.Ccp = (Amat @ self.Ccc @ Amat.T) + (self.Bmat @ S @ self.Bmat.T)


	def correct(self, observation):
		# Kalman gain
		# K = (Ccp @ self.Hmat.T) / ((self.Hmat @ Ccp @ self.Hmat.T) + self.sigmasq*self.kv)
		K = (self.Ccp @ self.Hmat.T) / ((self.Hmat @ self.Ccp @ self.Hmat.T) + self.kv)
		K = K.reshape(-1, 1)

		# correction step
		self.acc = self.acp + (K * (observation - self.Hmat @ self.acp))
		self.Ccc = self.Ccp - (K @ self.Hmat @ self.Ccp)

		# log prediction error decomposition to update particle weight
		self.logweight += self.log_ped(observation)
		

	def log_ped(self, observation):
		# Prediction Error Decomposition
		ayt = (self.Hmat @ self.acp)
		# Cyt = (self.Hmat @ Ccp @ self.Hmat.T) + (self.sigmasq*self.kv)
		Cyt = (self.Hmat @ self.Ccp @ self.Hmat.T) + (self.kv)
		Cyt = Cyt.flatten()

		# update log weight
		return -0.5 * np.log(2.*np.pi*Cyt) - (1./(2.*Cyt))*np.square(observation-ayt)


	def increment(self, observation, s, t):
		if type(s) == pd._libs.tslibs.timedeltas.Timedelta:
				s = s.total_seconds()
		if type(t) == pd._libs.tslibs.timedeltas.Timedelta:
				t = t.total_seconds()

		# kalman prediction step
		self.predict(s, t)

		# kalman correction step
		self.correct(observation)



class RBPF:
	"""
	Full rao-blackwellised (marginalised) particle filter
	"""
	def __init__(self, mumu, beta, kw, kv, theta, data, N, gsamps, epsilon):

		# x and y values for the timeseries
		self.times = data['Date_Time']
		self.prices = data['Price']
		self.nobservations = self.times.shape[0]

		# store initial values
		self.initial_time = self.times[0]
		self.initial_price = self.prices[0]

		# transform data so that first observation is zero and t=0
		self.prices = self.prices.subtract(self.initial_price)
		self.times = self.times.subtract(self.initial_time)

		# generators for passing through the times and observations
		self.timegen = iter(self.times)
		self.pricegen = iter(self.prices)

		self.prev_time = 0.
		self.prev_price = 0.
		self.current_time = next(self.timegen)
		self.current_price = next(self.pricegen)
		# implementation parameters
		# no. of particles
		self.N = N
		# limit for resampling based on effective sample size
		self.log_resample_limit = np.log(self.N*epsilon)
		self.log_marginal_likelihood = 0.

		# collection of particles
		self.particles = [LangevinParticle(mumu, beta, kw, kv, theta, gsamps) for _ in range(N)]
		

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

	
	def observe(self):
		# collect new times and prices
		self.prev_price = self.current_price
		self.current_price = next(self.pricegen)
		self.prev_time = self.current_time
		self.current_time = next(self.timegen)


	def increment_particles(self):
		"""
		Increment each particle based on the newest time and observation
		"""
		# reweight each particle -- could be faster using a map()?
		self.observe()
		for particle in self.particles:
			particle.increment(self.current_price, self.prev_time, self.current_time)


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


	def get_state_mean_pred(self):
		"""
		Get weighted sum of current particle means
		"""
		weights = np.array([np.exp(particle.logweight).reshape(1, -1) for particle in self.particles])
		means = np.array([particle.acp for particle in self.particles])
		return np.sum(weights*means, axis=0)


	def get_state_covariance(self):
		"""
		Get weighted sum of current particle variances
		"""
		weights = np.array([np.exp(particle.logweight).reshape(1, -1) for particle in self.particles])
		covs = np.array([particle.Ccc for particle in self.particles])
		return np.sum(weights*covs, axis=0)


	def get_state_covariance_pred(self):
		"""
		Get weighted sum of current particle variances
		"""
		weights = np.array([np.exp(particle.logweight).reshape(1, -1) for particle in self.particles])
		covs = np.array([particle.Ccp for particle in self.particles])
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


	def get_log_predictive_likelihood(self):
		lweights = np.array([particle.logweight for particle in self.particles])
		return logsumexp(lweights)


	def run_filter(self, ret_history=False):
		"""
		Main loop of particle filter
		"""
		if ret_history:
			state_means = [0.]
			state_variances = [0.]
			grad_means = [0.]
			grad_variances = [0.]

		for _ in tqdm(range(self.nobservations-1)):
			self.increment_particles()
			# log marginal term added before reweighting (based on predictive weight)
			self.log_marginal_likelihood += self.get_log_predictive_likelihood()
			self.reweight_particles()
			if ret_history:
				smean = self.get_state_mean()
				svar = self.get_state_covariance()
				state_means.append(smean[0, 0])
				state_variances.append(svar[0, 0])
				grad_means.append(smean[1, 0])
				grad_variances.append(svar[1, 1])

			if self.get_logDninf() < self.log_resample_limit:
				self.resample_particles()
		if ret_history:
			return np.array(state_means), np.array(state_variances), np.array(grad_means), np.array(grad_variances), self.log_marginal_likelihood
		else:
			return self.log_marginal_likelihood


	def run_filter_MP(self, theta):
		"""
		run_filter function slightly adjusted to be used for multiprocessing
		"""
		self.theta=theta

		for _ in (range(self.nobservations-1)):
			self.increment_particles()
			# log marginal term added before reweighting (based on predictive weight)
			self.log_marginal_likelihood += self.get_log_predictive_likelihood()
			self.reweight_particles()

			if self.get_logDninf() < self.log_resample_limit:
				self.resample_particles()
	
			return self.log_marginal_likelihood


	def run_filter_full_hist(self):
		"""
		Main loop of particle filter
		"""
		states = np.zeros((self.nobservations, self.N))
		grads = np.zeros((self.nobservations, self.N))

		# weights = np.zeros((self.nobservations, self.N))
		# weights[0,:] = np.ones(self.N)
		for i in tqdm(range(self.nobservations-1)):
			self.increment_particles()
			self.reweight_particles()

			if self.get_logDninf() < self.log_resample_limit:
				self.resample_particles()

			curr_states = np.array([particle.acc[0,0] for particle in self.particles])
			states[i+1, :] = curr_states
			curr_grads = np.array([particle.acc[1,0] for particle in self.particles])
			grads[i+1, :] = curr_grads
			# curr_weights = np.array([particle.logweight for particle in self.particles]).flatten()
			# curr_weights -= np.min(curr_weights)
			# curr_weights /= np.max(curr_weights)
			# curr_weights = np.nan_to_num(curr_weights, nan=1.0)
			# weights[i+1, :] = curr_weights

		return states, grads