import numpy as np
from functions import gen_poisson_epochs, sort_jumps
from tqdm import tqdm
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt

# desperately need to vectorise langevin_S -- currently langevin_S does 120its/sec, langevin_m does 20,000its/sec ....

def langevin_m(alpha, theta, V, c, deltat, Epochs):
	"""
	calculate the m vector for the langevin model
	"""
	# is alpha = C*t ???
	vec1 = np.array([[1./theta],[1.]])
	vec2 = np.array([[-1./theta],[0.]])
	# print(np.exp(theta*(t-V)).shape)
	# print(vec1.shape)
	t = c*deltat
	# only accept epochs up to some truncated level
	epochs_trunc = np.where(Epochs<t, Epochs, np.inf)
	summand = np.power(epochs_trunc, -1)*(np.exp(theta*(t-V))*vec1 + vec2)
	return alpha*deltat*np.sum(summand, axis=1)

def langevin_S(alpha, theta, V, c, deltat, Epochs):
	"""
	calculate the S vector for the langevin model
	"""
	t = c*deltat
	mat1 = np.array([[1./(theta**2),1./theta],[1./theta,1]])
	mat2 = np.array([[2./(theta**2),-1./theta],[-1./theta,0]])
	mat3 = np.array([[1./(theta**2),0],[0,0]])
	# would quite like to vectorise this
	# hard because need to mutiply a 4x4 array by an array of scalar values - output should be an array of 4x4's
	tot = 0
	epochs_trunc = np.where(Epochs<t, Epochs, np.inf)
	for i in range(epochs_trunc.shape[0]):
		term1 = np.exp(2*theta*(t-V[i]))*mat1
		term2 = np.exp(theta*(t-V[i]))*mat2
		term3 = mat3
		tot += np.power(epochs_trunc[i], -1)*(term1 + term2 + term3)
	return alpha*deltat*tot


def build_mat_exp(theta, dt):
	return np.array([[1., (np.exp(theta*dt)-1.)/theta],[0., np.exp(theta*dt)]])


def build_A(mat_exp, m_term):
	"""
	build the A matrix for the kalman filter
	"""
	zeros = np.zeros((1, 2))
	one = np.ones((1, 1))
	return np.block([[mat_exp, m_term],[zeros, one]])


def build_B(P):
	"""
	build the B matrix for the kalman filter
	"""
	v1 = np.eye(P)
	v2 = np.zeros((1, P))
	return np.block([[v1],[v2]])


def build_a00(P, muw):
	"""
	initial kalman a
	"""
	vec = np.zeros(P+1)
	vec[-1] = muw
	return vec


def build_C00(P, kw):
	"""
	initial kalman C tilde
	"""
	return np.block([[np.zeros((P, P)),np.zeros((P, 1))],[np.zeros((1, P)),np.array([kw])]])


def build_H(P):
	"""
	observation matrix
	"""
	vec = np.zeros(P+1)
	vec[0] = 1
	return vec


def kalman_predict(preva_up, prevC_up, Amat, Bmat, Ce):
	newa = np.matmul(Amat, preva_up)
	newC = np.matmul(Amat, prevC_up, Amat.T) + np.matmul(Bmat, Ce, Bmat.T)
	return newa, newC


def kalman_update(preva_pr, prevC_pr, Hmat, kv):
	yt = np.matmul(Hmat, state) + np.sqrt(sw_sq*kw)*np.random.randn(size=1)
	Kmat = prevC_pr/(np.matmul(Hmat, prevC_pr, Hmat)+kv)
	newa = preva_pr + (yt-np.matmul(Hmat, preva_pr))*Kmat
	newC = prevC_pr - np.matmul(Kmat, Hmat, prevC_pr)
	return newa, newC


T = 1.
th = 0.0001
dt = 1./1000.
C = 10. # for gamma process
mw = 1.
sws = 1.

def generate_langevin_ss(muw, theta, dt, C, T, esamps):
	# currently can't start at t = 0 -- does this mean that using rate 1/t is wrong for the poisson?
	t = 0
	# number of samps fully defined by final time and timestep
	samps = int(T/dt)
	# initial state is a Gaussian rv
	Xnew = multivariate_normal.rvs(mean=build_a00(1, muw), cov=build_C00(1, 1.)).T
	# matrix exponential for deterministic component
	A1 = build_mat_exp(theta, dt)
	# container for state skeleton
	Xs = np.zeros((samps, 2))
	for i in tqdm(range(samps)):
		# store value
		Xs[i] = Xnew
		Xold = Xnew
		# generate a series of poisson epochs up to time t -- I think this is probably wrong
		e = gen_poisson_epochs(1./(t+dt), esamps)
		# jump times
		v = T*np.random.rand(esamps)
		# calculate the langevin m and S, -- is the choice of C*t as alpha right?
		# t/dt is the truncation parameter
		m = langevin_m(C*t, theta, v, t/dt, dt, e) 
		S = langevin_S(C*t, theta, v, t/dt, dt, e)
		# process step using the calculate variables
		Xnew = np.matmul(A1, Xold) + Xold[1]*m + np.sqrt(sws)*multivariate_normal.rvs(mean=np.zeros(2), cov=S)
		t += dt
	return Xs


# print(langevin_m(C*1., th, v, 2./dt, dt, e))
# print(langevin_S(C*1., th, v, 2./dt, dt, e))

seq = generate_langevin_ss(mw, th, dt, C, T, 1000)
plt.plot(np.arange(seq.shape[0]), seq[:, 0])
plt.show()