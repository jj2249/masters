from process import *
import matplotlib.pyplot as plt

lss = LangevinModel(mu=0., sigmasq=100., beta=0.1, kv=0., theta=-0.5, nobservations=1000)
lss.forward_simulate()
plt.plot(lss.observationtimes, lss.observationvals)
plt.show()