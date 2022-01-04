import numpy as np
from scipy.stats import gamma

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

		

	def construct_timeseries(self):
		"""
		Construct a skeleton process on a uniform discrete time axis
		"""
		axis = np.linspace(self.minT, self.maxT, self.samps)
		cumulative_jumps = np.cumsum(self.jsizes)
		timeseries = np.zeros(self.samps)

		for i in range(1, self.samps):
			occured_jumps = self.jtimes[self.jtimes<axis[i]]
			if occured_jumps.size == 0:
				timeseries[i] = 0
			else:
				jump_idx = np.argmax(occured_jumps)
				timeseries[i] = cumulative_jumps[jump_idx]
		return axis, timeseries


	def plot_timeseries(self, ax):
		"""
		Plot the process skeleton
		"""
		t, f = self.construct_timeseries()
		ax.step(t, f)
		return ax


class GammaProcess(JumpProcess):
	def __init__(self, C, beta, samps=1000, minT=0., maxT=1.):
		JumpProcess.__init__(self, samps=samps, minT=minT, maxT=maxT)
		self.C = C
		self.beta = beta
		self.jtimes = self.generate_times()


	def generate_times(self):
		jtimes = np.linspace(self.minT, self.maxT, self.samps)
		return jtimes


	def generate(self):
		dt = self.jtimes[1]-self.jtimes[0]
		jumps = gamma.rvs(a=dt*self.C, loc=0, scale=1./self.beta, size=self.samps)
		self.jsizes = jumps


	def construct_timeseries(self):
		return self.jtimes, np.cumsum(self.jsizes)


	def marginal_gamma(self, x, t, ax, label=''):
		"""
		Plot the marginal gamma distribution on a given set of axes
		"""
		ax.plot(x, gamma.pdf(x, self.C*t, scale=1./self.beta), label=label)


	def marginal_gamma_cdf(self, x, t, ax, label=''):
		"""
		Plot the marginal gamma distribution on a given set of axes
		"""
		ax.plot(gamma.cdf(x, self.C*t, scale=1./self.beta), x, label=label)


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
	def __init__(self, C, beta, theta, sigmasq, samps=1000, minT=0., maxT=1., jtimes=None, epochs=None):
		JumpProcess.__init__(self, samps=samps, minT=minT, maxT=maxT, jtimes=jtimes, epochs=epochs)

		self.W = GammaProcess(C, beta, samps=samps, minT=minT, maxT=maxT)
		self.W.generate()

		# self.jtimes = self.generate_times()
		self.jtimes = self.W.jtimes

		self.theta = theta
		self.sigmasq = sigmasq
		self.sigma = np.sqrt(sigmasq)


	def generate(self):
		normal = np.random.randn(self.W.jsizes.shape[0]-1)
		self.jsizes = np.zeros(self.W.jsizes.shape[0])
		for i in range(1, self.W.jsizes.shape[0]):
			self.jsizes[i] = self.theta*self.W.jsizes[i-1]+self.sigma*np.sqrt(self.W.jsizes[i-1])*normal[i-1]


class LangevinModel:
	def __init__(self, muw, sigmasq, kv, theta, C, beta, nobservations):
		self.theta = theta
		self.nobservations = nobservations
		self.observationtimes = np.cumsum(np.random.exponential(scale=.1, size=nobservations))
		self.observationvals = []
		# initial state
		self.state = np.array([0, 0, muw])
		self.beta = beta
		self.C = C
		self.kv = kv
		self.sigmasq = sigmasq
		self.muw = muw

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
		Z = GammaProcess(self.C, self.beta, samps=1000, minT=self.s, maxT=self.t)
		Z.generate()
		m = Z.langevin_m(self.t, self.theta).reshape(-1, 1)
		S = self.sigmasq*Z.langevin_S(self.t, self.theta)
		Sc = np.linalg.cholesky(S + 1e-12*np.eye(2))
		Amat = self.A_matrix(Z, m, self.t-self.s)
		e = Sc @ np.random.randn(2)
		self.state = Amat @ self.state + self.Bmat @ e
		new_observation = self.Hmat @ self.state + np.sqrt(self.sigmasq*self.kv)*np.random.randn()
		self.observationvals.append(new_observation[0])
		
	
	def forward_simulate(self):
		for _ in range(self.nobservations-1):
				self.increment_process()
				self.s = self.t
				self.t = self.tgen.__next__()
		self.observationtimes = self.observationtimes[:-1]
		self.observationvals = np.array(self.observationvals)







