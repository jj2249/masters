import numpy as np

class Process:
	def __init__(self, samps=1000, minT=0., maxT=1.):
		# implementation parameters
		self.samps = samps
		self.rate = 1./(maxT-minT)
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

		return accepted_values[accepted_values>0.]


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

		for i in range(self.samps):
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


	def generate(self):
		# Overflow allowed for large Gamma and small c since large jumps are rejected wp 1
		old_settings = np.seterr()
		np.seterr(over='ignore')

		# jump sizes using the epochs
		x = 1./(self.beta*(np.exp(self.epochs/self.C)-1))
		np.seterr(**old_settings)

		a = (1+self.beta*x) * np.exp(-self.beta*x)

		self.jsizes = self.accept_samples(x, a)
		self.jtimes = self.generate_times(acc_samps=self.jsizes.shape[0])
		self.sort_jumps()


	def marginal_gamma(self, x, t, ax):
		"""
		Plot the marginal gamma distribution on a given set of axes
		"""
		ax.plot(x, gamma.pdf(x, self.C*t, scale=1/self.beta))


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
		return np.sum(np.array([[(vec2-2*vec1+1)/np.square(theta), vec3],
			[vec3, vec2]]), axis=2)


class VarianceGammaProcess(JumpProcess):
	def __init__(self, C, beta, mu, sigmasq, samps=1000, minT=0., maxT=1., jtimes=None, epochs=None):
		JumpProcess.__init__(self, samps=samps, minT=minT, maxT=maxT, jtimes=jtimes, epochs=epochs)

		self.W = GammaProcess(C, beta, samps=self.samps, minT=self.minT, maxT=self.maxT)
		self.W.generate()

		# self.jtimes = self.generate_times()
		self.jtimes = self.W.jtimes

		self.mu = mu
		self.sigmasq = sigmasq
		self.sigma = np.sqrt(sigmasq)


	def generate(self):
		normal = np.random.randn(self.W.jsizes.shape[0]-1)
		self.jsizes = np.zeros(self.W.jsizes.shape[0])
		for i in range(1, self.W.jsizes.shape[0]):
			self.jsizes[i] = self.mu*self.W.jsizes[i-1]+self.sigma*np.sqrt(self.W.jsizes[i-1])*normal[i-1]


	# def marginal_variance_gamma(self, x, t, ax):
	# 	"""
	# 	NEEDS WORK
	# 	"""
	# 	gamma_param = np.sqrt(2*self.beta)
	# 	nu_param = self.C*t
	# 	alpha_param = np.sqrt(betahat**2 + gamma_param**2)

	# 	term1 = np.power(gamma_param, 2*nu_param) * np.power(alpha_param, 1-2*nu_param)
	# 	term2 = np.sqrt(2*np.pi)*gamma_func(nu_param)*np.power(2, nu_param-1)
	# 	term3 = beta*np.abs(x-mu)
	# 	term4 = kv(nu_param-0.5, term3)
	# 	term5 = np.exp(betahat*(x-mu))

	# 	axes.plot(x, (term1/term2) * np.power(term3, nu_param-0.5) * term4 * term5)


class LangevinModel:
	def __init__(self, muw, kv, theta, C, beta, nobservations):
		self.theta = theta
		self.nobservations = nobservations
		self.observationtimes = np.cumsum(np.random.exponential(scale=.1, size=nobservations))
		self.observationvals = []
		# initial state
		self.state = np.array([0, 0, muw])
		self.beta = beta
		self.C = C
		self.kv = kv

		self.Bmat = self.B_matrix()
		self.Hmat = self.H_matrix()

		self.tgen = (time for time in self.observationtimes)
		self.s = 0
		self.t = self.tgen.__next__()


	def A_matrix(self, Z, m, dt):
		return np.block([[Z.langevin_drift(dt, self.theta), m.reshape(-1, 1)],
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
		m = Z.langevin_m(self.t, self.theta)
		S = Z.langevin_S(self.t, self.theta)
		Sc = np.linalg.cholesky(S + 1e-12*np.eye(2))
		Amat = self.A_matrix(Z, m, self.t-self.s)

		e = Sc @ np.random.randn(2)
		self.state = Amat @ self.state + self.Bmat @ e

		new_observation = self.Hmat @ self.state + np.sqrt(self.kv)*np.random.randn()
		self.observationvals.append(new_observation[0])
		
	
	def forward_simulate(self):
		for _ in range(self.nobservations-1):
				self.increment_process()
				self.s = self.t
				self.t = self.tgen.__next__()
		self.observationtimes = self.observationtimes[:-1]
		self.observationvals = np.array(self.observationvals)







