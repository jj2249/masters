import numpy as np
import copy
from src.process import GammaProcess, LangevinModel
import pandas as pd
from tqdm import tqdm


def logsumexp(lw, h, x, axis=0, retlog=False):
	"""
	Helper function for calculating the log of a sum of exponentiated values in a numerically stable way
	"""
	# c = (np.min(lw)+np.max(lw))*0.5
	c = np.max(lw)
	broad_l = np.broadcast_to((lw-c).flatten(), x.T.shape).T
	if retlog:
		return c + np.log(np.sum(np.exp(broad_l) * h(x), axis=axis))
	else:
		return np.exp(c) * np.sum(np.exp(broad_l) * h(x), axis=axis)


def twotermlogsumexp(a, b):
	c = max(a, b)
	return c + np.log(np.exp(a-c)+np.exp(b-c))


def logcumsum(lw):
	vals = []
	s= -np.inf
	for num in lw:
		s = twotermlogsumexp(s, num)
		vals.append(s)
	return np.array(vals)


class LangevinParticle(LangevinModel):
	"""
	Underlying particle object in the particle filter
	"""
	def __init__(self, mux, mumu, beta, kw, kv, kmu, rho, eta, theta, p, initial_observation):
		# model parameters
		self.theta = theta
		self.kv = kv
		self.kw = kw
		self.beta = beta
		self.rho = rho
		self.eta = eta
		
		# # implementation parameters
		# self.gsamps = gsamps

		# initial kalman parameters -- also form the prior distribution
		# a current current
		# C current current
		self.acc = np.array([mux, 0., mumu]).reshape(-1, 1)
		self.Ccc = kw*np.eye(3)

		Cc = np.linalg.cholesky(self.Ccc)

		self.acc = self.acc + (Cc @ np.random.randn(3)).reshape(-1, 1)
		# LangevinModel.__init__(self, initial_state[0], initial_state[1], initial_state[2], 1., beta, kv, kmu, theta, gsamps)
		LangevinModel.__init__(self, self.acc[0,0], self.acc[1,0], self.acc[2,0], 1., beta, kv, kmu, theta, p)
		# sample initial state using cholesky decomposition
		# Cc = np.linalg.cholesky(self.Ccc + 1e-12*np.eye(3))
		# self.alpha = self.acc + Cc @ np.random.randn(3)

		# log particle weight
		self.Hmat = self.H_matrix()
		self.Bmat = self.B_matrix()
		# self.logweight = self.get_initial_weight(initial_observation)
		self.logweight = 0.
		self.E = 0.
		self.count = 0.


	def __repr__(self):
		return str(#"alpha: "+self.alpha.__repr__()+'\n'
			"acc: "+self.acc.__repr__()+'\n'
			+"Ccc: "+self.Ccc.__repr__()+'\n'
			+"Un-normalised weight: "+str(np.exp(self.logweight))
			)


	def predict(self, s, t, ret=False):
		# time interval between two observations
		dt = t - s
		# latent gamma process
		Z = GammaProcess(1., self.beta, minT=s, maxT=t)
		Z.generate()

		# parameters for estimating stochastic integral
		try:
			S = self.langevin_S(t, self.theta, Z)
			_ = np.linalg.cholesky(S)
			m = self.langevin_m(t, self.theta, Z)
		except np.linalg.LinAlgError:
			S = np.zeros((2, 2))
			m = np.zeros((2, 1))
		Amat = self.A_matrix(m, dt)
		Ce = self.dynamical_noise_cov(S, dt)

		# prediction step
		self.acp = (Amat @ self.acc).reshape(-1, 1)
		self.Ccp = (Amat @ self.Ccc @ Amat.T) + (self.Bmat @ Ce @ self.Bmat.T)
		if ret:
			return self.acp, self.Ccp


	def log_weight_update(self, observation):
		self.count += 1
		ayt = (self.Hmat @ self.acp).item()
		Cyt = ((self.Hmat @ self.Ccp @ self.Hmat.T) + self.kv).item()
		# print(ayt, Cyt)
		prevE = self.E
		self.E += np.square(observation - ayt)/Cyt
		return ((-0.5*np.log(Cyt)) - (self.rho + (self.count/2.))*np.log(self.eta + self.E/2) + (self.rho + ((self.count-1.)/2.))*np.log(self.eta + prevE/2.)).item()
	

	def update_weight(self, observation):
		# self.logweight += self.log_ped(observation)
		self.logweight += self.log_weight_update(observation)


	def correct(self, observation):
		# Kalman gain
		# K = (Ccp @ self.Hmat.T) / ((self.Hmat @ Ccp @ self.Hmat.T) + self.sigmasq*self.kv)
		K = (self.Ccp @ self.Hmat.T) / ((self.Hmat @ self.Ccp @ self.Hmat.T) + self.kv)
		K = K.reshape(-1, 1)

		# correction step
		self.acc = self.acp + (K * (observation - self.Hmat @ self.acp))
		self.Ccc = self.Ccp - (K @ self.Hmat @ self.Ccp)

		# log prediction error decomposition to update particle weight
		self.update_weight(observation)


	def increment(self, observation, s, t):
		"""
		Kalman prediction and correction plus weight update
		"""
		# kalman prediction step
		self.predict(s, t)

		# kalman correction step
		self.correct(observation)


class RBPF:
	"""
	Full rao-blackwellised (marginalised) particle filter
	"""
	def __init__(self, mux, mumu, beta, kw, kv, kmu, rho, eta, theta, p, data, N, epsilon):

		# x and y values for the timeseries
		# self.times = data['Telapsed']
		# self.prices = data['Price']
		self.times = data['DateTime']
		self.prices = data['Bid']
		self.nobservations = self.times.shape[0]

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
		# self.logF = 0.

		self.theta = theta
		self.beta = beta
		self.kv = kv
		self.rho = rho
		self.eta = eta

		self.p = p

		self.mux = mux
		self.mumu = mumu

		# collection of particles
		self.particles = [LangevinParticle(mux, mumu, beta, kw, kv, kmu, rho, eta, theta, p, self.current_price) for _ in range(N)]
		self.normalise_weights()
	

	def normalise_weights(self):
		"""
		Renormalise particle weights to sum to 1
		"""
		lweights = np.array([particle.logweight for particle in self.particles]).reshape(-1, 1)
		# numerically stable implementation
		sum_weights = logsumexp(lweights, lambda x : 1., np.ones(lweights.shape[0]), retlog=True)
		for particle in self.particles:
			# log domain
			particle.logweight = particle.logweight - sum_weights
		return sum_weights

	
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


	def predict_particles(self, t_current, t_pred):
		for particle in self.particles:
			particle.predict(t_current, t_current+t_pred)


	def resample_particles(self, underflows=False):
		"""
		Resample particles using multinomial distribution, then set weights to 1/N
		"""

		lweights = np.array([particle.logweight for particle in self.particles]).flatten()
		# normalised weights are the probabilities
		weights = np.exp(lweights)
		if underflows:
			n_underflows = np.count_nonzero(np.isnan(weights))
		probabilites = np.nan_to_num(weights)

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
		if underflows:
			return n_underflows


	def log_resample_particles(self):
		lweights = np.array([particle.logweight for particle in self.particles]).flatten()
		new_particles = []
		q = logcumsum(lweights)
		for i in range(self.N):
			logu = np.log(np.random.rand())
			s = -np.inf
			j = np.min(np.where(q>=logu))
			new_particles.append(copy.copy(self.particles[j]))
			new_particles[-1].logweight = -np.log(self.N)
		self.particles = new_particles


	def get_state_posterior(self):
		"""
		Get the parameters of the corrected mixture distribution
		"""
		lweights = np.array([(particle.logweight) for particle in self.particles]).reshape(-1, 1)
		eX = np.array([particle.acc for particle in self.particles])

		msum = logsumexp(lweights, lambda x : x, eX, axis=0, retlog=False)

		eXXt = np.array([particle.Ccc + (particle.acc @ particle.acc.T) for particle in self.particles])

		# return msum, (np.sum(weights*eXXt, axis=0) - (msum @ msum.T))
		return msum, logsumexp(lweights, lambda x : x, eXXt, axis=0, retlog=False) - msum @ msum.T


	def get_state_posterior_predictive(self):
		"""
		Get the parameters of the predictive mixture distribution
		"""

		lweights = np.array([(particle.logweight) for particle in self.particles]).reshape(-1, 1)
		eX = np.array([particle.acp for particle in self.particles])

		msum = logsumexp(lweights, lambda x : x, eX, axis=0, retlog=False)

		eXXt = np.array([particle.Ccp + (particle.acp @ particle.acp.T) for particle in self.particles])

		return msum, (logsumexp(lweights, lambda x : x, eXXt, axis=0, retlog=False) - msum @ msum.T)


	def get_logPn2(self):
		"""
		Inverse sum of squares for estimating ESS
		"""
		lweights = np.array([particle.logweight for particle in self.particles])
		return -logsumexp(2*lweights, lambda x : 1., np.ones(lweights.shape[0]), retlog=True)


	def get_logDninf(self):
		"""
		Inverse maximum weight for estimating ESS
		"""
		lweights = np.array([particle.logweight for particle in self.particles])
		return -np.max(lweights)


	# def get_log_predictive_likelihood(self):
	# 	"""
	# 	Sum of predictive weights gives the log likelihood
	# 	"""
	# 	lweights = np.array([particle.logweight for particle in self.particles])
	# 	return logsumexp(lweights, lambda x : 1., np.ones(lweights.shape[0]), retlog=True)


	def sigma_posterior(self, x, count, E):
		rhod = self.rho + (count/2.)
		etad = self.eta + (E/2.)
		return -(rhod+1)*np.log(x)-np.divide(etad, x)


	def run_filter(self, log_resample=False, ret_history=False, plot_marginal=False, ax=None, axsamps=1000, smin=0.1, smax=15., tpred=0., sample=False, progbar=False):
		"""
		Main loop of particle filter
		"""
		if ret_history:
			MSEs = []
			dss = [np.log(self.N)]
			pss = [np.log(self.N)]
			state_means = []
			state_variances = []

			grad_means = []
			grad_variances = []

			mu_means = []
			mu_variances = []

			smean, svar = self.get_state_posterior()

			state_means.append(smean[0, 0])
			state_variances.append(svar[0, 0])

			grad_means.append(smean[1, 0])
			grad_variances.append(svar[1, 1])

			mu_means.append(smean[2, 0])
			mu_variances.append(svar[2, 2])

		if sample:
			smean, svar = self.get_state_posterior()
			state_samp = [smean + np.linalg.cholesky(svar) @ np.random.randn(3).reshape(-1, 1)]

		for _ in tqdm(range(self.nobservations-1), disable=not progbar):
			self.increment_particles()
			# log marginal term added before reweighting (based on predictive weight)
			incremental_log_like = self.normalise_weights()
			self.log_marginal_likelihood += incremental_log_like
			d = self.get_logDninf()
			p = self.get_logPn2()
			if d < self.log_resample_limit:
				if log_resample:
					self.log_resample_particles()
				else:
					self.resample_particles()
		

			if ret_history:
				spmean, spvar = self.get_state_posterior_predictive()
				smean, svar = self.get_state_posterior()

				MSEs.append((spmean[0, 0] - self.current_price)**2) # add mean square predictive error

				state_means.append(smean[0, 0])
				state_variances.append(svar[0, 0])

				grad_means.append(smean[1, 0])
				grad_variances.append(svar[1, 1])

				mu_means.append(smean[2, 0])
				mu_variances.append(svar[2, 2])

				dss.append(d)
				pss.append(p)

			if sample:
				smean, svar = self.get_state_posterior()
				# try:
					# chol = np.linalg.cholesky(svar)
				# except np.linalg.LinAlgError:
					# print(svar)
					# chol = np.linalg.cholesky(svar + 1e-10*np.eye(3))
				chol = np.linalg.cholesky(svar)
				state_samp.append(smean + chol @ np.random.randn(3).reshape(-1, 1))
		if ret_history and tpred>0.:
			self.times = self.times.tolist()
			self.times.append(self.prev_time+tpred)
			self.predict_particles(self.prev_time, tpred)

			# smean = self.get_state_mean_pred()
			# svar = self.get_state_covariance_pred()
			smean, svar = self.get_state_posterior_predictive()

			state_means.append(smean[0, 0])
			state_variances.append(svar[0, 0])

			grad_means.append(smean[1, 0])
			grad_variances.append(svar[1, 1])

			mu_means.append(smean[2, 0])
			mu_variances.append(svar[2, 2])

		if plot_marginal and ret_history:
			# weights = np.array([np.exp(particle.logweight) for particle in self.particles])
			lweights = np.array([particle.logweight for particle in self.particles])
			Es = np.array([particle.E for particle in self.particles])
			count = int(self.particles[0].count)
			mixture = np.zeros(axsamps)
			mode = 0.
			mean = 0.
			axis = np.linspace(smin, smax, axsamps)
			
			E = logsumexp(lweights, lambda x : x, Es, retlog=False)
			mixture = self.sigma_posterior(axis, count, E)

			mode = (self.eta + E/2.)/(self.rho+count/2.+1.)
			mean = (self.eta + E/2.)/(self.rho+count/2.-1.)

			ax.plot(axis, mixture-logsumexp(mixture, lambda x : 1., np.zeros(mixture.shape[0]), retlog=False))

			return np.array(state_means), np.array(state_variances), np.array(grad_means), np.array(grad_variances), np.array(mu_means), np.array(mu_variances), self.log_marginal_likelihood, ax, mode, mean, dss, pss, MSEs
		
		elif ret_history:
			return np.array(state_means), np.array(state_variances), np.array(grad_means), np.array(grad_variances), np.array(mu_means), np.array(mu_variances), self.log_marginal_likelihood, dss, pss, MSEs
		
		elif sample:
			return state_samp, self.log_marginal_likelihood

		else:
			return self.log_marginal_likelihood


	def run_filter_MP(self):
		"""
		run_filter function slightly adjusted to be used for multiprocessing
		"""
		for _ in (range(self.nobservations-1)):
			self.increment_particles()
			# log marginal term added before reweighting (based on predictive weight)
			incremental_log_like = self.normalise_weights()
			self.log_marginal_likelihood += incremental_log_like
			if self.get_logDninf() < self.log_resample_limit:
				self.resample_particles()
	
		return (self.theta, self.beta, self.log_marginal_likelihood)


	def run_filter_kv(self):
		for _ in (range(self.nobservations-1)):
			self.increment_particles()
			# log marginal term added before reweighting (based on predictive weight)
			incremental_log_like = self.normalise_weights()
			self.log_marginal_likelihood += incremental_log_like
			if self.get_logDninf() < self.log_resample_limit:
				self.resample_particles()
	
		return (self.kv, self.log_marginal_likelihood)


	def run_filter_full_hist(self, progbar=False):
		"""
		Run the particle filter and return all particles
		"""
		states = np.zeros((self.nobservations, self.N))
		grads = np.zeros((self.nobservations, self.N))
		skews = np.zeros((self.nobservations, self.N))
		weights = np.zeros((self.nobservations, self.N))
		weights[0, :] = -np.log(self.N)*np.ones(self.N)
		states[0, :] = np.array([particle.acc[0,0] for particle in self.particles])
		grads[0, :] = np.array([particle.acc[1,0] for particle in self.particles])
		skews[0, :] = np.array([particle.acc[2,0] for particle in self.particles])
		for i in tqdm((range(self.nobservations-1)), disable=not progbar):
			self.increment_particles()
			_ = self.normalise_weights()

			if self.get_logDninf() < self.log_resample_limit:
				self.resample_particles()

			curr_states = np.array([particle.acc[0,0] for particle in self.particles])
			states[i+1, :] = curr_states
			curr_grads = np.array([particle.acc[1,0] for particle in self.particles])
			grads[i+1, :] = curr_grads
			curr_skews = np.array([particle.acc[2,0] for particle in self.particles])
			skews[i+1, :] = curr_skews
			curr_weights = np.array([particle.logweight for particle in self.particles])
			weights[i+1, :] = curr_weights

		return states, grads, skews, weights

	def run_filter_full_predictive(self, npred):
		"""
		Run the particle filter and return all particles
		"""
		states = np.zeros((self.nobservations-npred+1, self.N))
		grads = np.zeros((self.nobservations-npred+1, self.N))
		skews = np.zeros((self.nobservations-npred+1, self.N))
		weights = np.zeros((self.nobservations-npred+1, self.N))
		weights[0, :] = -np.log(self.N)*np.ones(self.N)
		for i in tqdm(range(self.nobservations-1-npred)):
			self.increment_particles()
			_ = self.normalise_weights()

			if self.get_logDninf() < self.log_resample_limit:
				self.resample_particles()

			curr_states = np.array([particle.acc[0,0] for particle in self.particles])
			states[i+1, :] = curr_states
			curr_grads = np.array([particle.acc[1,0] for particle in self.particles])
			grads[i+1, :] = curr_grads
			curr_skews = np.array([particle.acc[2,0] for particle in self.particles])
			skews[i+1, :] = curr_skews
			curr_weights = np.array([particle.logweight for particle in self.particles])
			weights[i+1, :] = curr_weights


		predtimes = self.times.tolist()[:-npred]
		predtimes.append(self.times.iloc[-1])
		# print(predtimes)
		self.predict_particles(self.prev_time, predtimes[-1]-self.prev_time)
		curr_states = np.array([particle.acp[0,0] for particle in self.particles])
		states[-1, :] = curr_states
		curr_grads = np.array([particle.acp[1,0] for particle in self.particles])
		grads[-1, :] = curr_grads
		curr_skews = np.array([particle.acp[2,0] for particle in self.particles])
		skews[-1, :] = curr_skews
		curr_weights = np.array([particle.logweight for particle in self.particles])
		weights[-1, :] = curr_weights

		return predtimes, states, grads, skews, weights