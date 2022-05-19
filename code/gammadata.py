import numpy as np
from time import time
import src.process as proc
import matplotlib.pyplot as plt
from sys import argv

beta = float(argv[1])
# mu = 7.
# sigmasq = 1.
g = proc.GammaProcess(1., beta, minT=0., maxT=20.)

acceps = 0
final_vals = []
n = 100_000
t0 = time()
for i in range(n):
	gi = proc.GammaProcess(1., beta)
	acceps += gi.generate(ret_accs=True)
	final_vals.append(np.sum(gi.jsizes))
t1 = time()

gsamps = int(10./beta)
if gsamps < 50:
	gsamps = 50
elif gsamps > 10000:
	gsamps = 10000
# acc_rate = 100.*acceps/(n*gsamps)

s = 1_000_000
x0 = np.min(final_vals)
x1 = np.max(final_vals)
d = (x1-x0)/s
x = np.linspace(x0, x1, s)

totals, bins, _ = plt.hist(final_vals, density=True, bins=100)
plt.close()
broll = np.roll(bins, 1)
broll[0] = 0.
centres = ((bins+broll)/2)[1:]
# mse = np.mean(np.square(g.marginal_pdf(centres, 1.)-totals))

print("Beta: "+str(beta))
print("Gs: "+str(gsamps))
print("Acceptance rate (%): " + str(100*acceps/(n*gsamps)))
print("Mean square error (e-06): "+str(np.mean(np.square(g.marginal_pdf(centres, 1.)-totals))/1e-6))
print("Avg time per process (microsecs): " + str((t1-t0)/(n*1e-6)))