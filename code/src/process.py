import numpy as np
from scipy.stats import gamma
from scipy.special import kv
from scipy.special import gamma as gammaf
from scipy.integrate import quad

class Process:
	"""
	Most basic object for continuous time process - store:

		- no. of samples
		- start and end time
	"""
	def __init__(self, samps=1000, minT=0., maxT=1.):
		# implementation parameters
		self.samps = samps
		self.minT = minT
		self.maxT = maxT

class JumpProcess(Process):
	"""
	Extension of Process for processes which are pure jump -- store:

		- time of jumps
		- size of jumps
	"""
	def __init__(self, samps=1000, minT=0., maxT=1.):
		Process.__init__(self, samps=samps, minT=minT, maxT=maxT)
		self.jtimes = None
		self.jsizes = None

	
	def generate_times(self, no_of_acceps=None):
		"""
		Uniformly sample the jump times in the appropriate interval
		"""
		# if no number of acceptances provided, assume no rejections
		if no_of_acceps == None:
			no_of_acceps = self.samps

		# uniform rvs in [minT, maxT)
		times = (self.maxT-self.minT) * np.random.rand(no_of_acceps) + self.minT
		return times


	def accept_samples(self, values, probabilites):
		"""
		Given a set of values and associated acceptance probabilities, perform accept/reject step
		"""
		# uniform samples in [0, 1)
		uniform = np.random.rand(values.shape[0])
	
		# accept if the pval is higher than the corresponding uniform value
		accepted_values = np.where(probabilites>uniform, values, 0.)
		
		# remove any zeros
		return accepted_values[accepted_values>0.]


	def sort_jumps(self):
		"""
		Sort process jumps into chronological order
		"""
		# indices of sorted times
		idx = np.argsort(self.jtimes)

		# store times and jump sizes sorted in this order
		self.jtimes = np.take(self.jtimes, idx)
		self.jsizes = np.take(self.jsizes, idx)

		
	def construct_timeseries(self, samples):
		"""
		Construct a skeleton process on a uniform discrete time axis -- could this be made faster???
		"""
		# axis for visualisation
		axis = np.linspace(self.minT, self.maxT, samples)

		# accumulating sum of jump sizes
		cumulative_jumps = np.cumsum(self.jsizes)
		# container for output values
		timeseries = np.zeros(samples)

		# for each point on the time axis
		for i in range(1, samples):
			# extract jumps which have occured by the current time
			occured_jumps = self.jtimes[self.jtimes<axis[i]]
			if occured_jumps.size == 0:
				# if there are no jumps, then the o/p must be zero
				timeseries[i] = 0
			else:
				# otherwise, the o/p is the sum of all occured jumps
				jump_idx = np.argmax(occured_jumps)
				timeseries[i] = cumulative_jumps[jump_idx]
		return axis, timeseries


	def plot_timeseries(self, ax, samples=1000, label=''):
		"""
		Construct and then plot the process skeleton
		"""
		t, f = self.construct_timeseries(samples=samples)
		ax.step(t, f, label=label, lw=1.2)
		return ax


class GammaProcess(JumpProcess):
	"""
	Object for generating and storing a single realisation of the gamma process, plus related functions -- store:

		- model parameters
		- implementation parameters
		- latent epochs
	"""
	def __init__(self, alpha, beta, minT=0., maxT=1.):

		# mean drift rate and variance rate
		self.alpha = alpha
		self.beta = beta

		gsamps = int(10./self.beta)
		if gsamps < 50:
			gsamps = 50
		elif gsamps > 10000:
			gsamps = 10000
			print('Warning ---> beta too low for a good approximation')
		# print(gsamps)
		JumpProcess.__init__(self, samps=gsamps, minT=minT, maxT=maxT)

		# parameters for rejection sampling (directly from levy measure)
		self.C = alpha**2/beta
		self.B = alpha/beta
		self.rate = 1./(maxT-minT)

	
	def generate_epochs(self):
		"""
		Poisson epochs control the jump sizes
		"""
		# sum of exponential random variables
		# rate is dependent on the length of the interval spanned by the process
		times = np.random.exponential(scale=self.rate, size=self.samps)
		return np.cumsum(times)
	

	def generate(self, ret_accs=False):
		"""
		Rejection sampling method
		"""
		self.epochs = self.generate_epochs()
		
		# jump sizes
		xs = 1./(self.B*(np.exp(self.epochs/self.C)-1))
		# acceptance probabilities
		ps = (1+self.B*xs)*np.exp(-self.B*xs)

		# accept/reject step -- also remove any jumps with size zero
		self.jsizes = self.accept_samples(xs, ps)
		# corresponding jump times are uniformly distributed
		self.samps = self.jsizes.shape[0]
		self.jtimes = self.generate_times(no_of_acceps=self.samps)
		# sort jumps in ascending time order
		self.sort_jumps()
		if ret_accs:
			return self.samps


	def marginal_pdf(self, x, t):
		return gamma.pdf(x, a=t*self.alpha**2/self.beta, loc=0, scale=self.beta/self.alpha)


	def marginal_cdf(self, x, t):
		return gamma.cdf(x, a=t*self.alpha**2/self.beta, loc=0, scale=self.beta/self.alpha)


	def marginal_gamma(self, x, t, ax, label=''):
		"""
		Plot the marginal gamma distribution on a given set of axes
		"""
		ax.plot(x, self.marginal_pdf(x, t), label=label)


	def marginal_gamma_cdf(self, x, t, ax, label=''):
		"""
		Plot the marginal gamma cumulative distribution on a given set of axes
		"""
		ax.plot(x, self.marginal_cdf(x, t), label=label)


class VarianceGammaProcess(JumpProcess):
	"""
	Object for constructing and storing a single realisation of the VG process, plus it's latent gamma process
	"""
	def __init__(self, beta, mu, sigmasq, minT=0., maxT=1.):
		# generate latent gamma process
		# defined in terms of alpha (mu), beta (nu) parameterisation - note that alpha=1 always
		self.W = GammaProcess(1., beta, minT=minT, maxT=maxT)
		self.W.generate()

		# total number of samples can vary due to the rejection sampling
		samps = self.W.jtimes.shape[0]

		# parent intialisation
		JumpProcess.__init__(self, samps=samps, minT=minT, maxT=maxT)

		# VG jumps occur at the same time as the gamma jumps --- is this true???
		self.jtimes = self.W.jtimes

		# model parameters
		self.mu = mu
		self.sigmasq = sigmasq
		self.beta = beta


	def generate(self, ret_accs=False):
		"""
		Brownian motion with time steps taken as the jump sizes of the latent gamma process
		"""
		# faster to sample all normal rvs at once
		normal = np.random.randn(self.W.samps)
		self.jsizes = (self.mu*self.W.jsizes) + (np.sqrt(self.sigmasq*self.W.jsizes) * normal)
		if ret_accs:
			return self.W.samps


	def marginal_pdf(self, x, t):
		"""
		Marginal pdf of the VG process at time t, using the Bessel function
		"""
		term1 = 2*np.exp(self.mu*x/self.sigmasq)
		term2 = np.power(self.beta, t/self.beta)*np.sqrt(2*np.pi*self.sigmasq)*gammaf(t/self.beta)
		term3 = np.abs(x)/np.sqrt(2*self.sigmasq/self.beta + self.mu**2)
		term4 = (t/self.beta) - 0.5
		term5 = (1./self.sigmasq) * np.sqrt(self.mu**2 + (2*self.sigmasq/self.beta))*np.abs(x)

		return (term1/term2)*np.power(term3, term4) * kv(term4, term5)		

	def marginal_variancegamma(self, x, t, ax, label=''):
		"""
		Plot the marginal pdf of the VG process on a supplied set of axes
		"""
		ax.plot(x, self.marginal_pdf(x, t), label=label)


class LangevinModel:
	"""
	State-space langevin model. State vector contains position, derivative and mean reversion parameter
	"""
	def __init__(self, x0, xd0, mu, sigmasq, beta, kv, kmu, theta, p):
		# implementation paramters
		# self.gsamps = gsamps

		# model parameters
		self.theta = theta
		self.beta = beta
		self.kv = kv
		self.sigmasq = sigmasq
		self.kmu = kmu
		self.p = p

		# initial state
		self.state = np.array([x0, xd0, mu])
		# containers for state variables over time --- not final
		self.underlyingvals = [x0]
		self.observationvals = [x0]
		self.observationgrad = [xd0]
		self.observationmus = [mu]
		self.jtimes = []
		self.jsizes = []
		
		# use constructor functions to build state space matrices (only those that do not depend on generation of the non-linear part)
		self.Bmat = self.B_matrix()
		self.Hmat = self.H_matrix()


	def A_matrix(self, m, dt):
		"""
		State transition matrix constructor -- depends on non-linear part of the state (W)
		"""
		# num = np.random.rand()
		A = np.block([[self.langevin_drift(dt, self.theta), m],
						[np.zeros((1, 2)), 1.]])
		return A
		# if num < self.p:
		# 	A[1,1] = 0.
		# 	return A
		# else:
		# 	return A

	def B_matrix(self):
		"""
		Noise matrix in state space model
		"""
		return np.vstack([np.eye(2),
						np.zeros((1, 2))])
		# return np.eye(3)


	def H_matrix(self):
		"""
		Observation matrix in state space model
		"""
		h = np.zeros((1, 3))
		h[:, 0] = 1.
		return h
	

	def langevin_drift(self, dt, theta):
		"""
		e^(Adt) for langevin form of A -- i.e. deterministic part of solution
		"""
		return np.array([[1., (np.exp(theta*dt)-1.)/theta],
						 [0., np.exp(theta*dt)]])


	def langevin_m(self, t, theta, W):
		"""
		Mean vector of I(ft) for langevin A
		"""
		vec2 = np.exp(theta*(t - W.jtimes))
		vec1 = (vec2-1.)/theta
		return np.sum(np.array([vec1 * W.jsizes,
							vec2 * W.jsizes]), axis=1).reshape(-1, 1)


	def langevin_S(self, t, theta, W):
		"""
		Covariance matrix for I(ft) for langevin A
		"""
		vec1 = np.exp(theta*(t - W.jtimes))
		vec2 = np.square(vec1)
		vec3 = (vec2-vec1)/theta
		return np.sum(np.array([[W.jsizes*(vec2-2.*vec1+1.)/np.square(theta), W.jsizes*vec3],
			[W.jsizes*vec3, W.jsizes*vec2]]), axis=2)


	def dynamical_noise_cov(self, Smat, dt, reg=0.):
		"""
		Ce matrix for noise -- mu is uncorrelated with the main process
		"""
		# return np.block([[Smat+reg*np.eye(2), np.zeros(2).reshape(-1,1)],
		# 				[np.zeros(2).reshape(-1,1).T , dt*self.kmu]])
		return Smat + reg*np.eye(2)


	def increment_process(self):
		# latent jumps from gamma process
		Z = GammaProcess(1., self.beta, minT=self.s, maxT=self.t)
		Z.generate()

		# calculate m, S for this realisation of latent jumps
		m = self.langevin_m(self.t, self.theta, Z)
		S = self.sigmasq*self.langevin_S(self.t, self.theta, Z)

		# cholesky decomposition for sampling of noise
		Ce = self.dynamical_noise_cov(S, self.t-self.s, reg=0.)
		try:
			Cec = np.linalg.cholesky(Ce)
			e = Cec @ np.random.randn(2)
		except np.linalg.LinAlgError:
			# print('Truncating innovation to zero, dt = ' + str(self.t-self.s))
			e = np.zeros(2)

		# extended state transition matrix
		Amat = self.A_matrix(m, self.t-self.s)

		# state increment
		self.state = Amat @ self.state + self.Bmat @ e

		# observation bits --- not final
		new_observation = self.Hmat @ self.state + np.sqrt(self.sigmasq*self.kv)*np.random.randn()
		lastobservation = new_observation[0]
		self.underlyingvals.append(self.state[0])
		self.observationvals.append(lastobservation)
		self.observationgrad.append(self.state[1])
		self.observationmus.append(self.state[2])
		self.jtimes = self.jtimes + Z.jtimes.tolist()
		self.jsizes = self.jsizes + Z.jsizes.tolist()
		
	
	def generate(self, nobservations=100):
		"""
		Generate the realisation of the state vector through time -- needs tweaking to be a bit more useful in terms of output
		"""
		# example times for testing - add one more time that will be truncated for ease
		self.observationtimes = np.cumsum(np.random.exponential(scale=.1, size=nobservations+1))

		# generator for passing through the observed times
		self.tgen = iter(self.observationtimes)
		# start at t=0
		self.s = 0.
		self.t = next(self.tgen)


		for i in range(nobservations):
				self.increment_process()
				self.s = self.t
				self.t = next(self.tgen)
		# discard unused observation
		self.observationtimes = np.roll(self.observationtimes, 1)
		self.observationtimes[0] = 0.
		self.observationvals = np.array(self.observationvals)
		self.observationgrad = np.array(self.observationgrad)
		self.observationmus = np.array(self.observationmus)

