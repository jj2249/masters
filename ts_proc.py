import numpy as np
import matplotlib.pyplot as plt
from sys import argv
from functions import *


def gen_ts_process(alpha, c, beta, rate, samps, maxT=1):
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