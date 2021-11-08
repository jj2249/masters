import numpy as np
from functions import gen_poisson_epochs
from tqdm import tqdm

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

	# only accept epochs up to some truncated level
	summand = np.power(np.where(Epochs<c*deltat, Epochs, 0), -1./alpha)*(np.exp(theta*(t-V))*vec1 + vec2)
	return T**(1./alpha) * np.sum(summand, axis=1)

def langevin_S(alpha,theta, V, T, Epochs):
	"""
	calculate the S vector for the langevin model
	"""
	mat1 = np.array([[1./(theta**2),1./theta],[1./theta,1]])
	mat2 = np.array([[2./(theta**2),-1./theta],[-1./theta,0]])
	mat3 = np.array([[1./(theta**2),0],[0,0]])
	# would quite like to vectorise this
	# hard because need to mutiply a 4x4 array by an array of scalar values - output should be an array of 4x4's
	tot = 0
	for i in range(V.shape[0]):
		term1 = np.exp(2*theta*(t-V[i]))*mat1
		term2 = np.exp(theta*(t-V[i]))*mat2
		term3 = mat3
		tot += np.power(Epochs[i], -1./alpha)*(term1 + term2 + term3)
	return T**(2./alpha) * tot

T = 1.
t = 1.
th = 0.5
dt = 1.
C = 10. # for gamma process
e = gen_poisson_epochs(1., 1000)
v = T*np.random.rand(1000)
c=10

def build_mat_exp(theta, dt):
	return np.array([[1, (np.exp(theta*dt)-1)/theta],[0, np.exp(theta*dt)]])


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


print(kalman_predict(build_a00(1, 0.), build_C00(1, 1.), build_A(build_mat_exp(th, dt), langevin_m(C*t, th, v, c, dt, e)), build_B(1), langevin_S(C*t, th, v, c, dt, e)))