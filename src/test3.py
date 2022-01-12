from process import *
import numpy as np
import matplotlib.pyplot as plt
nu = 1.
theta = -0.5

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
mtot1 = np.zeros(2)
mtot2 = np.zeros(2)
Stot1 = np.zeros((2, 2))
Stot2 = np.zeros((2, 2))
for i in range(100):
	g = GammaProcess(1./nu, 1./nu, samps=100000, minT=0., maxT=1.)
	g.generate()
	g.plot_timeseries(ax1)
	mtot1 += g.langevin_m(1., theta)
	Stot1 += g.langevin_S(1., theta)
for i in range(100):
	g = GammaProcess(1./nu, 1./nu, samps=1000, minT=0., maxT=1.)
	g.generate()
	g.plot_timeseries(ax2)
	mtot2 += g.langevin_m(1., theta)
	Stot2 += g.langevin_S(1., theta)

print(mtot1/100)
print(mtot2/100)
print(Stot1/100)
print(Stot2/100)
plt.show()
# print(g.langevin_m(1., theta))
# print(g.langevin_S(1., theta))
