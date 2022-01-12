from process import *
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma

nu = 10.
dt = 1./10000.
g = GammaProcess(1./nu, 1./nu, samps=10000, minT=0., maxT=1.)
g.generate()
# print(g.jsizes)
# print(g.jtimes)
# plt.hist(g.jsizes, density=True, bins=100)
plt.hist(gamma.rvs(a=dt/nu, loc=0, scale=nu/dt, size=10000), density=True)
t = np.linspace(0., 1., 10000)
plt.plot(t, gamma.pdf(t, a=dt/nu, loc=0, scale=nu))
plt.show()