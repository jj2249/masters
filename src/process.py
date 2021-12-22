import numpy as np

class Process:
	def __init__(self, samps=1000, maxT=1.):
		# implementation parameters
		self.samps = samps
		self.rate = 1./maxT
		self.maxT = maxT

class JumpProcess(Process):
	def __init__(self, samps=1000, maxT=1., jtimes=None, epochs=None):
		Process.__init__(self, samps=samps, maxT=maxT)
		
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
		times = np.random.exponential(self.rate, size=self.samps)
		return np.cumsum(times)

	
	def generate_times(self):
		"""
		Uniformly sample the jump times
		"""
		# uniform rvs in [0, maxT)
		times = self.maxT * np.random.rand(self.samps)
		return times


	def accept_samples(self, values, probabilites):
		"""
		Method for generation of certain processes
		"""
		# random samples to decide acceptance
		uniform = np.random.rand(values.shape[0])
		# accept if the probability is higher than the generated value
		return np.where(probabilites>values, values, 0)


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
		axis = np.linspace(0., self.maxT, self.samps)
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
	def __init__(self, C, beta, samps=1000, maxT=1.):
		JumpProcess.__init__(self, samps=samps, maxT=maxT)
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
		self.jtimes = self.generate_times()
		self.sort_jumps()

	def marginal_gamma(self, x, t, ax):
		"""
		Plot the marginal gamma distribution on a given set of axes
		"""
		ax.plot(x, gamma.pdf(x, self.C*t, scale=1/self.beta))


class VarianceGammaProcess(JumpProcess):
	def __init__(self, C, beta, mu, sigmasq, samps=1000, maxT=1., jtimes=None, epochs=None):
		JumpProcess.__init__(self, samps=samps, maxT=maxT, jtimes=jtimes, epochs=epochs)

		self.W = GammaProcess(C, beta, samps=self.samps, maxT=self.maxT)
		self.W.generate()

		# self.jtimes = self.generate_times()
		self.jtimes = self.W.jtimes

		self.mu = mu
		self.sigmasq = sigmasq
		self.sigma = np.sqrt(sigmasq)


	def generate(self):
		normal = np.random.randn(self.samps-1)
		self.jsizes = np.zeros(self.samps)
		for i in range(1, self.samps):
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


class ngLevyProcess(Process):
	def __init__(self, Z):
		# Z is the driving jump process
		self.Z = Z

		self.maxT = self.Z.maxT
		self.samps = self.Z.samps
		self.dt = self.maxT/self.samps

		self.jtimes = self.Z.jtimes
		self.epochs = self.Z.epochs


	def get_drift(self):
		"""
		Default to unity case
		"""
		return np.exp(self.dt)


	def get_jump_sum(self, t):
		idx = np.argwhere((self.jtimes > t) & (self.jtimes < t+self.dt))
		Vfilt = self.jtimes[idx]
		Zfilt = self.Z.jsizes[idx]

		return np.array(np.sum(Zfilt * np.exp(t+self.dt-Vfilt)))


	def construct_timeseries(self):
		drift = self.get_drift()
		axis = np.linspace(0., self.maxT, self.samps)
		timeseries = np.zeros((self.samps, drift.shape[1]))

		for i in range(1, self.samps):
			timeseries[i] = drift @ timeseries[i-1] + self.get_jump_sum(axis[i])

		return axis, timeseries


	def plot_timeseries(self, ax):
		t, f = self.construct_timeseries()
		for i in range(len(ax)):
			ax[i].plot(t, f[:,i])
		return ax



class LangevinModel(ngLevyProcess):
	def __init__(self, Z, theta):
		self.theta = theta
		ngLevyProcess.__init__(self, Z)


	def get_drift(self):
		return np.array([[1., (np.exp(self.theta*self.dt)-1.)/self.theta],
						 [0., np.exp(self.theta*self.dt)]])


	def get_jump_sum(self, t):
		idx = np.argwhere((self.jtimes > t) & (self.jtimes < t+self.dt))
		Vfilt = self.jtimes[idx]
		Zfilt = self.Z.jsizes[idx]

		vec2 = np.exp(self.theta * (t+self.dt-Vfilt))
		vec1 = (vec2 - 1.)/self.theta

		return np.array([np.sum(vec1*Zfilt),
						np.sum(vec2*Zfilt)])

