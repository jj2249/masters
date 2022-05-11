from ./src/process import VarianceGammaProcess, GammaProcess
import numpy as np
from scipy.stats import norm
import matplotlib as mpl
# mpl.use("pgf")
# mpl.rcParams.update({
#     "pgf.texsystem": "pdflatex",
#     'font.family': 'serif',
#     'text.usetex': True,
#     'pgf.rcfonts': False
# })
import matplotlib.pyplot as plt

plt.style.use('ggplot')
tw = 6.50127
th = 9.00177

s = 1_000_000
n = 10000

# vg = VarianceGammaProcess(0.1, 0., 1.)
# vg2 = VarianceGammaProcess
g = GammaProcess(1., 1.)

final_vals = []
# final_vals2 = []
for i in range(n):
	# vgi = VarianceGammaProcess(0.1, 0., 1.)
	# vgi.generate()
	# final_vals.append(np.sum(vgi.jsizes))
	gi = GammaProcess(1., 1.)
	gi.generate()
	final_vals.append(np.sum(gi.jsizes))

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
x0 = np.min(final_vals)
x1 = np.max(final_vals)
d = (x1-x0)/s
x = np.linspace(x0, x1, s)

# vg.marginal_variancegamma(x, 1., ax1)
# vg.marginal_variancegamma(x, 2., ax)
# vg.marginal_variancegamma(x, 3., ax)
g.marginal_gamma(x, 1., ax1)
g.marginal_gamma_cdf(x, 1., ax2)

# pdf = vg.marginal_pdf(x, 1.)
# ax1.plot(x, pdf, label=r'$\mathcal{V}$')
# ax2.plot(x, np.cumsum(pdf)*d)
ax1.hist(final_vals, density=True, bins=100)
# ax2.hist(final_vals, density=True, cumulative=True, bins=100)

ax1.set_xlabel(r'$v$')
# ax2.set_xlabel(r'$\gamma$')
# ax1.set_title(r'Marginal pdf')
# ax2.set_title(r'Marginal cdf'))
ax1.set_title(r'$f_\mathcal{V}(v; t, 0, 1, 0.1)$')
# ax2.set_title(r'$F_\Gamma(\gamma; t, \alpha, \beta)$')
# ax1.set_xlim(-6., 6.)
# ax1.plot(x, norm.pdf(x), label=r'$\mathcal{N}')
# ax1.legend(loc='upper right')
fig.set_size_inches(w=0.5*tw, h=th/3.)
plt.tight_layout()
plt.show()

# plt.savefig('../resources/figures/variancegammamarg5.pgf')
# final_vals = np.sort(final_vals)
# cdf1 = g.marginal_cdf(final_vals, 1.)
# cdf2 = np.flip(cdf1)
# i = np.arange(n)
# sterm = np.sum((2*i-1)*(np.log(cdf1)+np.log(1-cdf2)))/n
# # print(sterm)
# print(-n-sterm)