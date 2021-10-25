import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from functions import *


def marginal_gamma(x, t, c, beta, samps=1000):
	plt.plot(x, (np.power(beta, c*t))/(gamma(c*t))*np.power(x, (c*t)-1)*np.exp(-beta*x))


def gen_gamma_process(c, beta, rate, samps, maxT=1):
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

	gamma_process = np.cumsum(jumps)

	return times_sorted, gamma_process