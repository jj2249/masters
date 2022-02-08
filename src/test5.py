from process import *
import matplotlib.pyplot as plt

fig = plt.figure()
ax1 = fig.add_subplot(211)
ax2 = fig.add_subplot(212)

lss = LangevinModel(mu=0., sigmasq=1., beta=.8, kv=0., theta=-.5, nobservations=1000, gsamps=200)
lss.forward_simulate()
ax1.plot(lss.observationtimes, lss.observationvals)
ax2.plot(lss.observationtimes, lss.observationgrad, color='r')
ax2.set_xlabel('t')
ax2.set_ylabel('xdot')
ax1.set_ylabel('x')
plt.show()

### --- Different values of beta in the gamma process --- ###

# fig = plt.figure()
# ax = fig.add_subplot()

# g1 = GammaProcess(alpha=1., beta=1., samps=1_000_000)
# g1.generate()
# g1.plot_timeseries(ax, label='beta: 1.0')
# g2 = GammaProcess(alpha=1., beta=.1, samps=1_000_000)
# g2.generate()
# g2.plot_timeseries(ax, label='beta: 0.1')
# g3 = GammaProcess(alpha=1., beta=.01, samps=1_000_000)
# g3.generate()
# g3.plot_timeseries(ax, label='beta: 0.01')
# fig.legend()
# plt.show()