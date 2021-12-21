import numpy as np
import matplotlib.pyplot as plt
from process import VarianceGammaProcess
from statespacev2 import LangevinStateSpace


vg = VarianceGammaProcess(10., 0.1, 0., 1.)
vg.generate()

lss = LangevinStateSpace(vg, -0.1)
fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122)
lss.plot_timeseries(fig.axes)
plt.show()