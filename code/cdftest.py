from src.process import VarianceGammaProcess, GammaProcess
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
# mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False
})
import matplotlib.pyplot as plt
from time import time
from tqdm import tqdm

plt.style.use('ggplot')
tw = 6.50127
th = 9.00177

s = 1_000_000
n = 10000

# vg2 = VarianceGammaProcess
# betas = np.array([2., 1., 0.5, 0.1, 0.05, 0.01, 0.001])
alpha = 2.
beta = .1
mu = 0.
sigmasq = 1.
vg = VarianceGammaProcess(beta, mu, sigmasq)
# g = GammaProcess(alpha, beta, minT=0., maxT=1.)
final_vals = []
# final_vals2 = []
# gsamples = np.arange(100, 3000, 50)
# gsamples = np.array([2000])
# acceptance_rates = []
# mses = []
# avg_times = []

# acceps = 0
# t0 = time()
for i in range(n):
	vgi = VarianceGammaProcess(beta, mu, sigmasq)
	vgi.generate()
	final_vals.append(np.sum(vgi.jsizes))
	# gi = GammaProcess(alpha, beta)
	# gi.generate()
	# final_vals.append(np.sum(gi.jsizes))
# t1 = time()
# print("Avg time: " + str((t1-t0)/n))
# avg_times.append((t1-t0)/n)
# gsamps = int(10./beta)
# if gsamps > 10000:
	# gsamps = 10000
# elif gsamps < 50:
	# gsamps = 50
# print("Acceptance rate: " + str(100*acceps/(n*gsamps)))
# acceptance_rates.append(100*acceps/(n*gsamps))
fig = plt.figure()
ax1 = fig.add_subplot(111)
# ax2 = fig.add_subplot(122)
x0 = np.min(final_vals)
x1 = np.max(final_vals)
d = (x1-x0)/s
x = np.linspace(x0, x1, s)

# vg.marginal_variancegamma(x, 1., ax1)
# vg.marginal_variancegamma(x, 2., ax)
# vg.marginal_variancegamma(x, 3., ax)
# g.marginal_gamma(x, 1., ax1)
# g.marginal_gamma_cdf(x, 1., ax1)

pdf = vg.marginal_pdf(x, 1.)
ax1.plot(x, pdf, label=r'$\mathcal{V}$')
# ax1.plot(x, np.cumsum(pdf)*d)
totals, bins, _ = ax1.hist(final_vals, density=True, bins=100, histtype='stepfilled', alpha=0.5)
# plt.close()
# broll = np.roll(bins, 1)
# broll[0] = 0.

# centres = ((bins+broll)/2)[1:]
# print("Mean square error: "+str(np.mean(np.square(g.marginal_pdf(centres, 1.)-totals))))
# mses.append(np.mean(np.square(g.marginal_pdf(centres, 1.)-totals)))
# ax2.hist(final_vals, density=True, cumulative=True, bins=100)

ax1.set_xlabel(r'$v$')
# ax1.set_xlabel(r'$\gamma$')
# ax1.set_title(r'Marginal pdf')
# ax2.set_title(r'Marginal cdf'))
# ax1.set_title(r'$f_\mathcal{V}(v; t, 0, 1, 0.1)$'))
# ax2.set_title(r'$F_\Gamma(\gamma; t, \alpha, \beta)$')
# ax1.set_xlim(-6., 6.)
ax1.plot(x, norm.pdf(x), label=r'$\mathcal{N}$')
ax1.legend(loc='upper right')
# fig.set_size_inches(w=0.5*tw, h=th/3.)
# plt.tight_layout()
# plt.show()

# plt.savefig('../resources/figures/variancegammamarg5.pgf')

# final_vals = np.sort(final_vals)
# cdf1 = g.marginal_cdf(final_vals, 1.)
# cdf2 = np.flip(cdf1)
# i = np.arange(n)
# sterm = np.sum((2*i-1)*(np.log(cdf1)+np.log(1-cdf2)))/n
# # print(sterm)
# print(-n-sterm)

# plt.plot(gsamples, mses)
# plt.plot(gsamples, acceptance_rates)
fig.set_size_inches(0.5*tw, th/3.5)
fig.tight_layout()
plt.show()

# np.savez('./gammatimingdata_e1.npz', acceptance_rates, avg_times, mses, pickle=True)