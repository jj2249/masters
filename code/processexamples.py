from process import VarianceGammaProcess, GammaProcess
import numpy as np

import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False
})
import matplotlib.pyplot as plt

plt.style.use('ggplot')
tw = 6.50127

fig = plt.figure()
ax1 = fig.add_subplot(111)

betas = [0.1, 1., 1.5]
for beta in betas:
# for i in range(5):
	# vg = VarianceGammaProcess(0.001, 0., 1., gsamps=10000)
	# vg.generate()
	# vg.plot_timeseries(ax1)
	g = GammaProcess(1., beta, samps=10000)
	g.generate()
	g.plot_timeseries(ax1, label=r'$\beta = {}$'.format(beta))
# ax1.set_xlabel(r'$v$')
# ax2.set_xlabel(r'$v$')
# # ax1.set_title(r'Marginal pdf')
# # ax2.set_title(r'Marginal cdf')
# ax1.set_title(r'$f_\mathcal{V}(v; t, \mu, \sigma^2, \beta)$')
# ax2.set_title(r'$F_\mathcal{V}(v; t, \mu, \sigma^2, \beta)$')

fig.set_size_inches(w=1.*tw, h=9.*tw/16.)
ax1.set_xlabel(r'Time, $t$')
ax1.set_ylabel(r'$\Gamma_t$')
ax1.legend(loc='upper left')
plt.tight_layout()
# plt.show()
plt.savefig('../resources/figures/gammas.pgf')