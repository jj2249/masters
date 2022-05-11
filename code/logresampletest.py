import numpy as np
from src.particlefilter import logsumexp
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm
import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False
})
plt.style.use('seaborn')
tw = 6.50127
def twotermlogsumexp(a, b):
	c = max(a, b)
	return c + np.log(np.exp(a-c)+np.exp(b-c))


def logcumsum(lw):
	vals = []
	s= -np.inf
	for num in lw:
		s = twotermlogsumexp(s, num)
		vals.append(s)
	return np.array(vals)


def log_resample(nums):
	idx = []
	q = logcumsum(nums)
	for i in range(nums.shape[0]):
		logu = np.log(np.random.rand())
		s = -np.inf
		j = np.min(np.where(q>=logu))
		idx.append(j)
	return idx


def resample(nums):
	"""
	Resample particles using multinomial distribution, then set weights to 1/N
	"""
	probabilites = np.nan_to_num(nums)
	# need to renormalise to account for any underflow when exponentiating -- better way to do this?
	probabilites = probabilites / np.sum(probabilites)
	# multinomial method returns an array with the number of selections stored at each location
	selections = np.random.multinomial(nums.shape[0], probabilites)
	return selections

def time_func(f, N):
	t0 = time()
	f(N)
	t1 = time()
	return t1 - t0

tnorm = []
tlog = []
Ns = np.arange(10, 10000, 100)
for N in tqdm(Ns):
	nums = np.random.rand(N)
	lnums = np.log(nums)
	tnorm.append(1000.*time_func(resample, nums))
	tlog.append(1000.*time_func(log_resample, lnums))

fig = plt.figure()
ax = fig.add_subplot()
ax.plot(Ns, tnorm, label=r'\texttt{Numpy}')
ax.plot(Ns, tlog, label='Log-resampling')

ax.set_xlabel(r'Number of particles, $N$')
ax.set_ylabel(r'Execution time $(ms)$')
fig.set_size_inches(w=tw, h=0.5*tw)
ax.legend(loc='upper left')

plt.tight_layout()
# plt.savefig('../resources/figures/resamplingtimings.pgf')
# plt.savefig('../resources/figures/resamplingtimings.pdf')

plt.show()