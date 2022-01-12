from process import *
import matplotlib.pyplot as plt


fig = plt.figure()
ax = fig.add_subplot()
for _ in range(3):
	vg = VarianceGammaProcess(beta=0.1, mu=0., sigmasq=1., samps=10000)
	vg.generate()
	vg.plot_timeseries(ax)
ax.set_xlabel('t')
ax.set_ylabel('VGt')
plt.show()