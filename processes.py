import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from functions import *
from scipy.stats import gamma


def marginal_gamma(x, t, c, beta, axes, samps=1000):
	"""
	Plot the marginal gamma distribution on a given set of axes
	"""
	axes.plot(x, gamma.pdf(x, c*t, scale=1/beta))

def gamma_cdf(x, t, c, beta, axes, samps=1000):
	axes.plot(x, gamma.cdf(x, c*t, scale=1/beta))


def gen_gamma_process(c, beta, rate, samps, maxT=1):
	"""
	Generate a sample gamma process
	"""
	# generate a set of poisson epochs
	es = gen_poisson_epochs(rate, samps)
	
	# jump sizes - Overflow allowed for large Gamma and small c since large jumps are rejected wp 1
	old_settings = np.seterr()
	np.seterr(over='ignore')
	x = 1.0/(beta*(np.exp(es/c)-1))
	np.seterr(**old_settings)
	# acceptance probabilities: very small jumps get accepted wp ~1,
	# large jumps get rejected wp ~1
	acceps = (1+beta*x) * np.exp(-beta * x)
	# keep selected samples
	accepted = accept(x, acceps)

	# independently generated jump times
	times = maxT*np.random.rand(samps)
	#sort jumps
	times_sorted, jumps = sort_jumps(times, accepted)

	# return gamma process - SHOULD (0,0) be set as the first value??

	gamma_process = np.roll(np.cumsum(jumps), 1)
	gamma_process[0] = 0.0
	times_sorted = np.roll(times_sorted, 1)
	times_sorted[0] = 0.0

	return times_sorted, gamma_process

def gen_ts_process(alpha, c, beta, rate, samps, maxT=1):
	"""
	Generate a sample Tempered Stable process
	"""
	# generate a set of poisson epochs
	es = gen_poisson_epochs(rate, samps)
	
	# jump sizes - QUITE UNSTABLE ATM
	x = np.power(alpha*es/c, -1/alpha)
	# acceptance probabilities
	acceps = np.exp(-beta*x)
	# uniform samples for deciding whether to accept
	unis = np.random.rand(samps)
	# keep selected samples
	accepted = accept(x, acceps)

	# independently generated jump times
	times = maxT*np.random.rand(samps)
	#sort jumps
	times_sorted, jumps = sort_jumps(times, accepted)

	# return ts process - SHOULD (0,0) be set as the first value??
	ts_process = np.cumsum(jumps)

	return times_sorted, ts_process

def generate_brownian_motion(mu, sigma_sq, samps, maxT=1):
	"""
	Generate simple Brownian motion
	"""
	# times
	t = np.linspace(0, maxT, samps)
	# process skeleton
	X = np.zeros(samps)
	# step through process (could this be vectorised???)
	for i in range(1, samps):
		normal = np.random.randn(1)
		X[i] = X[i-1] + np.sqrt(sigma_sq*(t[i]-t[i-1]))*normal + mu*(t[i]-t[i-1])
	return t, X

def variance_gamma(mu, sigma_sq, gamma_proc, maxT=1, returnBM=False):
	"""
	Generate a sample from the variance gamma process given a gamma subordinator
	"""
	gamma_jumps = np.diff(gamma_proc)
	samps = gamma_jumps.shape[0]
	t = np.linspace(0, maxT, samps)
	if returnBM:
		B = np.zeros(samps)
	X = np.zeros(samps)
	for i in range(1, samps):
		normal = np.random.randn(1)
		if returnBM:
			B[i] = B[i-1] + np.sqrt(sigma_sq*(t[i]-t[i-1]))*normal + mu*(t[i]-t[i-1])
		X[i] = X[i-1] + np.sqrt((sigma_sq)*gamma_jumps[i-1])*normal + mu*gamma_jumps[i-1]

	# need to replace vg times with uniform jumps Vi
	# vg_times = np.linspace(0, maxT, samps)
	vg_times = T*np.linspace(samps)
	if returnBM:
		return t, B, vg_times, X
	else:
		return vg_times, X