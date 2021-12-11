import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from functions import *
from scipy.stats import gamma
from scipy.special import kv
from scipy.special import gamma as gamma_func


def marginal_gamma(x, t, c, beta, axes, samps=1000):
	"""
	Plot the marginal gamma distribution on a given set of axes
	"""
	axes.plot(x, gamma.pdf(x, c*t, scale=1/beta))


def marginal_variance_gamma(x, t, c, beta, mu, betahat, axes, samps=1000):
	gamma_param = np.sqrt(2*beta)
	nu_param = c*t
	alpha_param = np.sqrt(betahat**2 + gamma_param**2)

	term1 = np.power(gamma_param, 2*nu_param) * np.power(alpha_param, 1-2*nu_param)
	term2 = np.sqrt(2*np.pi)*gamma_func(nu_param)*np.power(2, nu_param-1)
	term3 = beta*np.abs(x-mu)
	term4 = kv(nu_param-0.5, term3)
	term5 = np.exp(betahat*(x-mu))

	axes.plot(x, (term1/term2) * np.power(term3, nu_param-0.5) * term4 * term5)


def variance_gamma_pdf(x, mu, alpha, beta, lam, samps=1000):
	gam = np.sqrt(alpha**2 - beta**2)
	term1 = np.power(gam, 2*lam) * np.power(np.abs(x-mu), lam-0.5) * kv(lam-0.5, alpha*np.abs(x-mu))
	term2 = np.sqrt(np.pi)*gamma_func(lam) * np.power(2*alpha, lam-0.5)
	term3 = np.exp(beta*(x-mu))
	return np.divide(term1, term2) * term3


def gamma_cdf(x, t, c, beta, axes, samps=1000):
	axes.plot(x, gamma.cdf(x, c*t, scale=1/beta))


def gen_gamma_process(c, beta, samps=1000, maxT=1, return_latents=False):
	"""
	Generate a sample gamma process
	"""
	# generate a set of poisson epochs
	es = gen_poisson_epochs(1./maxT, samps)
	
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
	if return_latents:
		return times_sorted, gamma_process, es
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

def variance_gamma(mu, sigma_sq, latent_times, gamma_proc, maxT=1):
	"""
	Generate a sample from the variance gamma process given a gamma subordinator
	latent times currently passed for clarity as this is the latent times of both the gamma subordinator and the variance gamma
	"""
	gamma_jumps = np.diff(gamma_proc)
	samps = gamma_jumps.shape[0]
	X = np.zeros(samps)
	for i in range(1, samps):
		normal = np.random.randn(1)
		X[i] = X[i-1] + np.sqrt((sigma_sq)*gamma_jumps[i-1])*normal + mu*gamma_jumps[i-1]
	return latent_times[1:], X