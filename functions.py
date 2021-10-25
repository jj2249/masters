import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def accept(x_values, acceptance_probs):
	# generate uniform samples for comparison
	uniform_samples = np.random.rand(x_values.shape[0])
	# perform the comparison
	return  np.array([x_values[i] if acceptance_probs[i] > uniform_samples[i] else 0 for i in range(uniform_samples.shape[0])])


def sort_jumps(times, jumps):
	# sort times into order
	indices = np.argsort(times)
	# sort jumps in the same order as their corresponding times
	times_sorted = np.sort(times)
	jumps_sorted = np.take_along_axis(jumps, indices, axis=None)
	return times_sorted, jumps_sorted


def generate_and_plot(process, process_samps, gamma_marginal=None):
	""" Generate samples from the process and plot with time, plus plot a histogram of the end times """
	end_points = np.zeros(process_samps)
	plt.subplot(121)
	for i in range(process_samps):
		t, g = process()
		end_points[i] = g[-1]
		plt.step(t, g)

	plt.subplot(122)
	if gamma_marginal is not None:
		gamma_marginal()
	plt.hist(end_points, bins=50, density=True)
	plt.show()


def gen_poisson_epochs(rate, samps):
	""" Generate poisson epochs with a fixed rate """
	times = np.random.exponential(rate, size=samps)
	return times.cumsum()