from process import *
import matplotlib.pyplot as plt

# beta = .1

# vg = VarianceGammaProcess(beta=beta, mu=0., sigmasq=1.)
# x = np.linspace(-3., 3., 1000)
# fig = plt.figure()
# ax = fig.add_subplot(211)
# ax2 = fig.add_subplot(212)

# vg.marginal_variancegamma(x, 1., ax, label='mu: '+str(-1.))
# vg2 = VarianceGammaProcess(beta=beta, mu=0., sigmasq=10.)
# vg3 = VarianceGammaProcess(beta=beta, mu=0., sigmasq=150.)
# vg2.marginal_variancegamma(x, 1., ax, label='mu: '+str(0.))
# vg3.marginal_variancegamma(x, 1., ax, label='mu: '+str(1.))
# ax.set_xlabel('VG1')
# ax.set_ylabel('frequency')
# fig.legend()
# plt.show()

# lss = LangevinModel(mu=0., sigmasq=1., beta=.8, kv=0., theta=-.5, nobservations=1000)
# lss.forward_simulate()
# ax.plot(lss.observationtimes, lss.observationvals)
# ax2.plot(lss.observationtimes, lss.observationgrad, color='r')
# ax2.set_xlabel('t')
# ax2.set_ylabel('xdot')
# ax.set_ylabel('x')
# plt.show()

fig = plt.figure()
ax = fig.add_subplot()

g1 = GammaProcess(alpha=1., beta=1., samps=1_000_000)
g1.generate()
g1.plot_timeseries(ax, label='beta: 1.0')
g2 = GammaProcess(alpha=1., beta=.1, samps=1_000_000)
g2.generate()
g2.plot_timeseries(ax, label='beta: 0.1')
g3 = GammaProcess(alpha=1., beta=.01, samps=1_000_000)
g3.generate()
g3.plot_timeseries(ax, label='beta: 0.01')
fig.legend()
plt.show()