import numpy as np
from functions import *
from tqdm import tqdm
from scipy.linalg import cholesky
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from processes import *

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


def langevin_update_vector(theta, t1, t2, V, dZ):
	Vfilt, dZfilt = filter_jumps_by_times(t1, t2, V, dZ)
	
	vect2 = np.exp(theta*(t1-Vfilt))
	vect1 = (vect2-1)/theta
	return np.array([np.sum(vect1*dZfilt), np.sum(vect2*dZfilt)])


def forward_langevin(c, beta, mu, sigma_sq, theta, samps=1000, maxT=1.):
	V, W, E = gen_gamma_process(c, beta, samps, maxT, return_latents=True)
	V2, Z = variance_gamma(mu, sigma_sq, V, W, maxT=maxT)
	dZ = np.diff(Z)
	V3 = V2[:-1]
	t = 0.
	dt = maxT/samps
	# initial state is a Gaussian rv
	Xcurr = np.zeros(2)
	# container for state skeleton
	Xs = np.zeros((samps, 2))
	eAdt = build_mat_exp(theta, dt)
	# stochastic sum term in update expression
	for i in range(samps):
		Xs[i] = Xcurr
		term1 = langevin_update_vector(th, t, t+dt, V3, dZ)
		term2 = np.matmul(eAdt, Xcurr)
		# print(term1, term2)
		Xcurr = term1 + term2
		t += dt
	return V3, Xs

	## for forward samples --> build matrix exp, generate initial state, need to be able to split V's and Z's according to time intervals
# print(langevin_m(C*1., th, v, 2./dt, dt, e))
# print(langevin_S(C*1., th, v, 2./dt, dt, e))


T = 1.
dt = 1./1000.

C = 5. # jump arrival rate - Gamma    ---- Large C seems to be causing problems... could it be back to the overflow issue
BETA = 1.5 # inverse jump size - Gamma

th = -5. # langevin theta < 0

mw = 0. # prior mean for initial state
kv = 1. # prior variance for initial (mean component) state

muvg = 0. # mu parameter in VG process
ssq_vg = 1. # variance parameter in VG process


fig, [ax1, ax2] = plt.subplots(ncols=2)
for i in tqdm(range(1)):
	Times, X = forward_langevin(C, BETA, muvg, ssq_vg, th, samps=1000)
	Xf = X[:-2, 0]# + 0.01*np.random.randn(X[:-2, 0].shape[0])
	Xd = X[:-2, 1]
	ax1.plot(Times, Xf, color='orange')	
	ax2.step(Times, Xd)
ax2.set_xlabel('t')
ax1.set_xlabel('t')
ax1.set_title('x')
ax2.set_title('xdot')
plt.show()