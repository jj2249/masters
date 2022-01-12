import numpy as np


def ksdensity(data, width=0.3):
	def ksd(x_axis):
		def n_pdf(x, mu=0., sigma=1.):
			u = (x - mu) / abs(sigma)
			y = (1 / (np.sqrt(2 * np.pi) * abs(sigma)))
			y *= np.exp(-u * u / 2)
			return y
		prob = [n_pdf(x_i, data, width) for x_i in x_axis]
		pdf = [np.average(pr) for pr in prob]
		return np.array(pdf)
	return ksd