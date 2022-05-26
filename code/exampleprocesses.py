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
tw = 6.50127
th = 9.00177
plt.style.use('ggplot')
fig = plt.figure()
ax1 = fig.add_subplot()
fig2 = plt.figure()
ax2 = fig2.add_subplot()
np.random.seed(22)
for i in range(3):
	# vg = VarianceGammaProcess(.1, 0., 1.)
	# vg.generate()
	# vg.plot_timeseries(ax1)
	g = GammaProcess(1., 0.1)
	g.generate()
	g.plot_timeseries(ax1)
np.random.seed(22)
for i in range(3):
	# vg1 = VarianceGammaProcess(.001, 0., 1.)
	# vg1.generate()
	# vg1.plot_timeseries(ax2)
	g1 = GammaProcess(1., 0.5)
	g1.generate()
	g1.plot_timeseries(ax2)

fig.set_size_inches(w=0.5*tw, h=th/3.)
fig2.set_size_inches(w=0.5*tw, h=th/3.)
ax1.set_xlabel(r'Time, $t$')
ax2.set_xlabel(r'Time, $t$')
fig.tight_layout()
fig2.tight_layout()
plt.show()