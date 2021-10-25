from functions import *
from gamma_proc import gen_gamma_process

C = 1.
BETA = 0.1
samps = 10000

MU = 0
SIGMA = 1

t, T = gen_gamma_process(C, BETA, 1, samps, maxT=1)

def generate_brownian_motion(mu, sigma_sq, samps, maxT=1):
	t = np.linspace(0, maxT, samps)
	X = np.zeros(samps)
	for i in range(1, samps):
		normal = np.random.randn(1)
		X[i] = X[i-1] + np.sqrt(sigma_sq*(t[i]-t[i-1]))*normal + mu*(t[i]-t[i-1])
	return t, X

gamma_steps = np.diff(T)


def variance_gamma(mu, sigma_sq, gamma_jumps, maxT=1):
	samps= gamma_jumps.shape[0]
	t = np.linspace(0, maxT, samps)
	B = np.zeros(samps)
	X = np.zeros(samps)
	for i in range(1, samps):
		normal = np.random.randn(1)
		B[i] = B[i-1] + np.sqrt(sigma_sq*(t[i]-t[i-1]))*normal + mu*(t[i]-t[i-1])
		X[i] = X[i-1] + np.sqrt(sigma_sq*gamma_jumps[i-1])*normal + mu*gamma_jumps[i-1]
	vg_times = np.linspace(0, maxT, samps)
	return t, B, vg_times, X

t1, B, t2, Y = variance_gamma(MU, SIGMA, gamma_steps)

t2min, step = np.linspace(0, 1, samps-1, retstep=True)

t2max = t2min+step

plt.subplot(131)
plt.title('Gamma Subordinator')
plt.step(t, T)
plt.subplot(132)
plt.title('Brownian Motion')
plt.step(t1, B)
plt.subplot(133)
plt.title('Deformed BM')
# plt.hlines(Y, t2min, t2max)
plt.step(t2, Y)
plt.show()