import numpy as np
import matplotlib.pyplot as plt

BETA = 1
C = 1
T = 1

def gen_poisson_epochs(rate, samps):
	G = np.zeros(samps)
	time = 0
	for i in range(samps):
		time += np.random.exponential(rate)
		G[i] = time

	return G

def gen_gamma_process(c, beta, rate, samps, maxT):

	es = gen_poisson_epochs(rate, samps)
	y = np.zeros(samps)
	"""plt.subplot(121)
	plt.scatter(es, y)
	plt.subplot(122)
	plt.hist(np.diff(es), bins=20)
	plt.show()"""

	x = 1.0/(beta*(np.exp(es/c)-1))
	acceps = (1+beta*x) * np.exp(-beta * x)
	unis = np.random.rand(samps)

	accepted = np.array([x[i] if acceps[i] > unis[i] else 0 for i in range(acceps.shape[0])])
	times = maxT*np.random.rand(samps)
	indices = np.argsort(times)
	times_sorted = np.sort(times)
	jumps = np.take_along_axis(accepted, indices, axis=None)

	gamma_process = np.cumsum(jumps)

	return times_sorted, gamma_process

end_points = np.zeros(100)
plt.subplot(121)
for i in range(100):
	t, g = gen_gamma_process(C, BETA, 1, 100, 1)
	end_points[i] = g[-1]
	plt.step(t, g)
plt.subplot(122)
plt.hist(end_points, bins=20)
plt.show()