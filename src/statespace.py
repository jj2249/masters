import numpy as np
from process import Process, VarianceGammaProcess

class StateSpaceProcess(Process):
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



class LangevinStateSpace(StateSpaceProcess):
	def __init__(self, Z, theta):
		self.theta = theta
		StateSpaceProcess.__init__(self, Z)


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

