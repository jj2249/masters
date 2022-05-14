import numpy as np
import matplotlib as mpl
# mpl.use("pgf")
import matplotlib.pyplot as plt
plt.style.use('ggplot')
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    # 'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False
})


data = np.load('./gammatimingdata.npz')

Gs = np.arange(5, 300)[:-160]
acc = data['arr_0'][:-160]
timing = data['arr_1'][:-160]
mse = data['arr_2'][:-160]

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(Gs, acc, label='Acc')
ax2 = ax.twinx()
ax2.plot(Gs, mse, label='MSE', color='cornflowerblue')
ax2.grid(None)
tw = 6.50127
th = 9.00177
fig.set_size_inches(w=1.*tw, h=th/4.)
ax.set_xlabel(r'$G_s$')
ax.set_ylabel(r'Acceptance rate (\%)')
ax2.set_ylabel(r'MSE')
plt.tight_layout()
fig.legend(loc='center right', bbox_to_anchor=(0.85, 0.55))
plt.show()