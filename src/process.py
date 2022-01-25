import numpy as np
from scipy.stats import gamma
from scipy.special import kv
from scipy.special import gamma as gammaf
from scipy.integrate import quad

class Process:
	def __init__(self, samps=1000, minT=0., maxT=1.):
		# implementation parameters
		self.samps = samps
		self.rate = 1./(maxT-minT)
		# self.rate = 1.
		self.minT = minT
		self.maxT = maxT

class JumpProcess(Process):
	def __init__(self, samps=1000, minT=0., maxT=1., jtimes=None, epochs=None):
		Process.__init__(self, samps=samps, minT=minT, maxT=maxT)
		
		# latent parameters
		if jtimes == None:
			self.jtimes = self.generate_times()
		else:
			self.jtimes = jtimes
		if epochs == None:
			self.epochs = self.generate_epochs()
		else:
			self.epochs = epochs

		# jump sizes determined by epochs
		self.jsizes = None

	
	def generate_epochs(self):
		"""
		Poisson epochs control the jump sizes
		"""
		# sum of exponential random variables
		times = np.random.exponential(scale=self.rate, size=self.samps)
		return np.cumsum(times)

	
	def generate_times(self, acc_samps=None):
		"""
		Uniformly sample the jump times
		"""
		if acc_samps == None:
			acc_samps = self.samps
		# uniform rvs in [minT, maxT)
		times = (self.maxT-self.minT) * np.random.rand(acc_samps) + self.minT
		return times


	def accept_samples(self, values, probabilites):
		"""
		Method for generation of certain processes
		"""
		# random samples to decide acceptance
		uniform = np.random.rand(values.shape[0])
		# accept if the probability is higher than the generated value
		accepted_values = np.where(probabilites>values, values, 0)

		# return accepted_values[accepted_values>0.]
		return accepted_values


	def sort_jumps(self):
		"""
		Sort the process jumps into time order
		"""
		# sort times into order
		idx = np.argsort(self.jtimes)
		# return times and jump sizes sorted in this order
		self.jtimes = np.take(self.jtimes, idx)
		self.jsizes = np.take(self.jsizes, idx)

		

	# def construct_timeseries(self):
	# 	"""
	# 	Construct a skeleton process on a uniform discrete time axis
	# 	"""
	# 	axis = np.linspace(self.minT, self.maxT, self.samps)
	# 	cumulative_jumps = np.cumsum(self.jsizes)
	# 	timeseries = np.zeros(self.samps)

	# 	for i in range(1, self.samps):
	# 		occured_jumps = self.jtimes[self.jtimes<axis[i]]
	# 		if occured_jumps.size == 0:
	# 			timeseries[i] = 0
	# 		else:
	# 			jump_idx = np.argmax(occured_jumps)
	# 			timeseries[i] = cumulative_jumps[jump_idx]
	# 	return axis, timeseries


	def construct_timeseries(self):
		return self.jtimes, np.cumsum(self.jsizes)


	def plot_timeseries(self, ax, label=''):
		"""
		Plot the process skeleton
		"""
		t, f = self.construct_timeseries()
		ax.step(t, f, label=label)
		return ax


class GammaProcess(JumpProcess):
	def __init__(self, alpha, beta, samps=1000, minT=0., maxT=1.):
		JumpProcess.__init__(self, samps=samps, minT=minT, maxT=maxT)
		self.alpha = alpha
		self.beta = beta
		self.jtimes = self.generate_times()


	def generate_times(self):
		jtimes = np.linspace(self.minT, self.maxT, self.samps)
		return jtimes


	def generate(self):
		dt = self.jtimes[1]-self.jtimes[0]
		jumps = gamma.rvs(a=dt*self.alpha**2/self.beta, loc=0, scale=self.beta/self.alpha, size=self.samps)
		self.jsizes = jumps


	def construct_timeseries(self):
		return self.jtimes, np.cumsum(self.jsizes)


	def marginal_gamma(self, x, t, ax, label=''):
		"""
		Plot the marginal gamma distribution on a given set of axes
		"""
		ax.plot(gamma.pdf(x, a=t*self.alpha**2/self.beta, loc=0, scale=self.beta/self.alpha), x, label=label)


	def marginal_gamma_cdf(self, x, t, ax, label=''):
		"""
		Plot the marginal gamma distribution on a given set of axes
		"""
		ax.plot(x, gamma.cdf(x, a=t*self.alpha**2/self.beta, loc=0, scale=self.beta/self.alpha), label=label)


	def langevin_drift(self, dt, theta):
		return np.array([[1., (np.exp(theta*dt)-1.)/theta],
						 [0., np.exp(theta*dt)]])


	def langevin_m(self, t, theta):
		vec2 = np.exp(theta*(t - self.jtimes))
		vec1 = (vec2-1.)/theta
		return np.sum(np.array([vec1 * self.jsizes,
							vec2 * self.jsizes]), axis=1)


	def langevin_S(self, t, theta):
		vec1 = np.exp(theta*(t - self.jtimes))
		vec2 = np.square(vec1)
		vec3 = (vec2-vec1)/theta
		return np.sum(np.array([[self.jsizes*(vec2-2*vec1+1)/np.square(theta), self.jsizes*vec3],
			[self.jsizes*vec3, self.jsizes*vec2]]), axis=2)


class VarianceGammaProcess(JumpProcess):
	def __init__(self, beta, mu, sigmasq, samps=1000, minT=0., maxT=1., jtimes=None, epochs=None):
		JumpProcess.__init__(self, samps=samps, minT=minT, maxT=maxT, jtimes=jtimes, epochs=epochs)

		self.W = GammaProcess(1, beta, samps=samps, minT=minT, maxT=maxT)
		self.W.generate()

		# self.jtimes = self.generate_times()
		self.jtimes = self.W.jtimes

		self.mu = mu
		self.sigmasq = sigmasq
		self.sigma = np.sqrt(sigmasq)
		self.beta = beta


	# def generate(self):
	# 	normal = np.random.randn(self.W.jsizes.shape[0]-1)
	# 	self.jsizes = np.zeros(self.W.jsizes.shape[0])
	# 	for i in range(1, self.W.jsizes.shape[0]):
	# 		self.jsizes[i] = self.mu*self.W.jsizes[i-1]+self.sigma*np.sqrt(self.W.jsizes[i-1])*normal[i-1]


	def generate(self):
		normal = np.random.randn(self.samps)
		self.jsizes = (self.mu*self.W.jsizes) + (np.sqrt(self.sigmasq*self.W.jsizes) * normal)


	def marginal_pdf(self, x, t):
		term1 = 2*np.exp(self.mu*x/self.sigmasq)
		term2 = np.power(self.beta, t/self.beta)*np.sqrt(2*np.pi*self.sigmasq)*gammaf(t/self.beta)
		term3 = np.abs(x)/np.sqrt(2*self.sigmasq/self.beta + self.mu**2)
		term4 = (t/self.beta) - 0.5
		term5 = (1./self.sigmasq) * np.sqrt(self.mu**2 + (2*self.sigmasq/self.beta))*np.abs(x)

		return (term1/term2)*np.power(term3, term4) * kv(term4, term5)		

	def marginal_variancegamma(self, x, t, ax, label=''):
		ax.plot(x, self.marginal_pdf(x, t), label=label)


class LangevinModel:
	def __init__(self, mu, sigmasq, beta, kv, theta, nobservations):
		self.theta = theta
		self.nobservations = nobservations
		self.observationtimes = np.cumsum(np.random.exponential(scale=.1, size=nobservations))
		self.observationvals = []
		self.observationgrad = []
		self.lastobservation = 0.
		# initial state
		self.state = np.array([0, 0, mu])
		self.beta = beta
		self.kv = kv
		self.sigmasq = sigmasq
		self.muw = mu

		self.Bmat = self.B_matrix()
		self.Hmat = self.H_matrix()

		self.tgen = (time for time in self.observationtimes)
		self.s = 0
		self.t = self.tgen.__next__()


	def A_matrix(self, Z, m, dt):
		return np.block([[Z.langevin_drift(dt, self.theta), m],
						[np.zeros((1, 2)), 1.]])


	def B_matrix(self):
		return np.vstack([np.eye(2),
						np.zeros((1, 2))])


	def H_matrix(self):
		h = np.zeros((1, 3))
		h[0] = 1.
		# can also observe the derivative
		# h[1] = 1.
		return h


	def increment_process(self):
		Z = GammaProcess(1., self.beta, samps=1000, minT=self.s, maxT=self.t)
		Z.generate()
		# m = self.lastobservation*Z.langevin_m(self.t, self.theta).reshape(-1, 1)
		# S = (self.lastobservation**2)*self.sigmasq*Z.langevin_S(self.t, self.theta)
		m = Z.langevin_m(self.t, self.theta).reshape(-1, 1)
		S = self.sigmasq*Z.langevin_S(self.t, self.theta)
		Sc = np.linalg.cholesky(S + 1e-12*np.eye(2))
		Amat = self.A_matrix(Z, m, self.t-self.s)
		e = Sc @ np.random.randn(2)
		self.state = Amat @ self.state + self.Bmat @ e
		new_observation = self.Hmat @ self.state + np.sqrt(self.sigmasq*self.kv)*np.random.randn()
		lastobservation = new_observation[0]
		self.observationvals.append(lastobservation)
		self.observationgrad.append(self.state[1])
		
	
	def forward_simulate(self):
		for _ in range(self.nobservations-1):
				self.increment_process()
				self.s = self.t
				self.t = self.tgen.__next__()
		self.observationtimes = self.observationtimes[:-1]
		self.observationvals = np.array(self.observationvals)







