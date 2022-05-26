from src.process import VarianceGammaProcess, GammaProcess
import numpy as np

import matplotlib as mpl
# mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False
})
import matplotlib.pyplot as plt

plt.style.use('ggplot')
tw = 6.50127


s = 1_000_000
n = 10000

vg = VarianceGammaProcess(1., 0., 1.)
vg2 = VarianceGammaProcess(2., 0., 1.)
vg3 = VarianceGammaProcess(1., 1., 1.)
# vg4 = VarianceGammaProcess()
g = GammaProcess(2., 1.)

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
x0 = -5.
x1 = 5.
d = (x1-x0)/s
x = np.linspace(x0, x1, s)

vg.marginal_variancegamma(x, 1., ax1)
vg2.marginal_variancegamma(x, 1., ax2)
# g.marginal_gamma(x, 1., ax1)

ax1.set_xlabel(r'$v$')
ax2.set_xlabel(r'$v$')
# ax1.set_title(r'Marginal pdf')
# ax2.set_title(r'Marginal cdf')
ax1.set_title(r'$f_\mathcal{V}(v; t, \mu, \sigma^2, \beta)$')
ax2.set_title(r'$F_\mathcal{V}(v; t, \mu, \sigma^2, \beta)$')

fig.set_size_inches(w=1.*tw, h=9.*tw/16.)
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