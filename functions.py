import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gamma

def accept(x_values, acceptance_probs):
	"""
	Sift a set of jump values for given acceptance probabilites
	"""
	# generate uniform samples for comparison
	uniform_samples = np.random.rand(x_values.shape[0])
	# perform the comparison
	return  np.array([x_values[i] if acceptance_probs[i] > uniform_samples[i] else 0 for i in range(uniform_samples.shape[0])])


def sort_jumps(times, jumps):
	"""
	Sort jumps into ascending time order (i.e not sorted by jump size)
	"""
	# sort times into order
	indices = np.argsort(times)
	# sort jumps in the same order as their corresponding times
	times_sorted = np.sort(times)
	jumps_sorted = np.take_along_axis(jumps, indices, axis=None)
	return times_sorted, jumps_sorted


def generate_and_plot(process, process_samps, axes, hlines=False):
	"""
	Generate samples from the process and plot with time, and return the end times
	"""
	end_points = np.zeros(process_samps)
	for i in range(process_samps):
		t, g = process()
		end_points[i] = g[-1]
		if hlines:
			samps = g.shape[0]
			tmin, step = np.linspace(0, 1, samps, retstep=True)
			tmax = tmin+step
			red = np.random.rand()
			green = np.random.rand()
			blue = np.random.rand()
			color = (red, green, blue)
			axes.hlines(g, tmin, tmax, color=color)
		else:
			axes.step(t, g)
	return end_points

def gen_poisson_epochs(rate, samps):
	"""
	Generate poisson epochs with a fixed rate
	"""
	times = np.random.exponential(rate, size=samps)
	return times.cumsum()

def moving_average(x, w):
	"""
	Simple moving average filter
	"""
	return np.convolve(x, np.ones(w), 'valid') / w