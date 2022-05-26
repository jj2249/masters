import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use("pgf")
mpl.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False
})

plt.style.use('ggplot')
tw = 6.50127
th = 9.00177


data = np.load('./samples4.npz', allow_pickle=True)
arr0 = data['arr_0']
arr1 = data['arr_1']

acceps = np.array([arr0[i][0] for i in range(8)])
betas = np.array([np.exp(arr0[i][1][250::10,0]) for i in range(8)]).flatten()
thetas = np.array([-1.*np.exp(arr0[i][1][250::10,1]) for i in range(8)]).flatten()

betas0 = np.exp(arr0[2][1][250::10,0]).flatten()
thetas0 = np.exp(arr0[2][1][250::10,1]).flatten()


fig1 = plt.figure()
ax1 = fig1.add_subplot()
ax1.hist(betas, density=True, bins=100, histtype='stepfilled')
ax1.vlines(x=1., ymin=0., ymax=1.7, ls='--', color='black')
ax1.set_xlabel(r'$\beta$')
fig1.set_size_inches(w=0.5*tw, h=th/3.5)


fig2 = plt.figure()
ax2 = fig2.add_subplot()
ax2.hist(thetas, density=True, bins=100, histtype='stepfilled')
ax2.vlines(x=-3., ymin=0., ymax=.81, ls='--', color='black')
ax2.set_xlabel(r'$\theta$')
ax2.set_xticks([-7, -6, -5, -4, -3, -2, -1])
fig2.set_size_inches(w=0.5*tw, h=th/3.5)

fig3 = plt.figure()
ax3 = fig3.add_subplot()
ax3.scatter(betas, thetas, s=0.5)

fig4 = plt.figure()
ax4 = fig4.add_subplot()
ax4.acorr(betas0, usevlines=False, maxlags=150, ms=1.5)
ax4.set_ylim(0.)
ax4.set_xlim(0., 100.)
ax4.set_xlabel('Lag')
fig4.set_size_inches(w=0.5*tw, h=th/3.5)

fig5 = plt.figure()
ax5 = fig5.add_subplot()
ax5.acorr(thetas0, usevlines=False, maxlags=150, ms=1.5)
ax5.set_ylim(0.)
ax5.set_xlim(0., 100.)
ax5.set_xlabel('Lag')
fig5.set_size_inches(w=0.5*tw, h=th/3.5)

fig6 = plt.figure()
ax6 = fig6.add_subplot(211)
axx6 = fig6.add_subplot(212)
ax6.plot(arr1.item().observationtimes, arr1.item().observationvals, color='red', ls='--', lw=1., marker='.', ms=2., mec='black', mfc='black')
axx6.plot(arr1.item().observationtimes, arr1.item().observationgrad, color='red', ls='--', lw=1., marker='.', ms=2., mec='black', mfc='black')
fig6.set_size_inches(w=1.*tw, h=th/3.5)
plt.setp(ax6.get_xticklabels(), visible=False)
ax6.set_ylabel(r'Position, $x$')
axx6.set_ylabel(r'Velocity, $\dot{x}$')
axx6.set_xlabel(r'Time, $t$')

print(np.mean(betas), np.mean(thetas))
fig1.tight_layout()
fig2.tight_layout()
fig3.tight_layout()
fig4.tight_layout()
fig5.tight_layout()
fig6.tight_layout()

# plt.show()
fig6.savefig('../resources/figures/mcmclss.pgf')
fig2.savefig('../resources/figures/mcmcthetahist.pgf')
fig1.savefig('../resources/figures/mcmcbetahist.pgf')
fig4.savefig('../resources/figures/mcmcbetacorr.pgf')
fig5.savefig('../resources/figures/mcmcthetacorr.pgf')
