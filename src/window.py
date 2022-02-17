import tkinter as tk
import numpy as np
import pandas as pd

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)
import matplotlib.pyplot as plt

from process import LangevinModel
from particlefilter import RBPF



### --- Forward Simulation --- ###


lss = LangevinModel(mu=0., sigmasq=1., beta=0.8, kv=0.1, theta=-0.8, gsamps=10_000)
lss.generate(nobservations=150)


## - store data in a dataframe - ##
sampled_dic = {'Date_Time': lss.observationtimes, 'Price': lss.observationvals}
sampled_data = pd.DataFrame(data=sampled_dic)



class Application(tk.Frame):
	def __init__(self, data, master=None):
		tk.Frame.__init__(self, master)
		self.createWidgets()
		self.rbpf = RBPF(mumu=0., beta=0.8, kw=1e6, kv=0.1, theta=-0.8, data=data, N=1_000, gsamps=5_000, epsilon=0.5)
		self.is_observe = True
		self.is_predict = False
		self.is_correct = False

	def createWidgets(self):
		fig = plt.figure(figsize=(20,12))
		ax = fig.add_subplot(111)
		canvas = FigureCanvasTkAgg(fig, master=window)
		canvas.get_tk_widget().grid(row=0, column=1)
		canvas.draw()

		frame = tk.Frame(window)
		frame.grid(row=10, column=0, rowspan=1, columnspan=9)
		toolbar = NavigationToolbar2Tk(canvas, frame)
		canvas.get_tk_widget().grid(row=1, column=0, rowspan=9, columnspan=9)
		toolbar.update()

		self.step_button = tk.Button(master=window,
					command=lambda: self.step(ax, canvas, self.rbpf),
					height=2,
					width=10,
					text="Step")

		self.step_button.grid(row=0, column=0, columnspan=9)


	def step(self, ax, canvas, rbpf):
		if self.is_observe:
			self.observe(ax, canvas, rbpf)
			self.is_observe = False
			self.is_predict = True
		elif self.is_predict:
			self.predict(ax, canvas, rbpf)
			self.is_predict = False
			self.is_correct = True
		elif self.is_correct:
			self.correct(ax, canvas, rbpf)
			self.is_correct = False
			self.is_observe = True


	def observe(self, ax, canvas, rbpf):
		rbpf.observe()
		ts = np.array([rbpf.prev_time, rbpf.current_time])
		ps = np.array([rbpf.prev_price, rbpf.current_price])
		try:
			ax.lines.pop(-2)
			# ax.collections.pop(-2)
		except:
			pass
		ax.plot(ts, ps, c='black')
		canvas.draw()


	def predict(self, ax, canvas, rbpf):
		aprev = rbpf.get_state_mean()
		cprev = rbpf.get_state_covariance()

		for particle in rbpf.particles:
			particle.predict(rbpf.prev_time, rbpf.current_time)

		ts = np.array([rbpf.prev_time, rbpf.current_time])
		pm = np.array([aprev[0,0], rbpf.get_state_mean_pred()[0,0]])
		pv = np.array([cprev[0,0], rbpf.get_state_covariance_pred()[0,0]])

		self.p = ax.plot(ts, pm, c='red', ls='--')
		# self.bounds = ax.fill_between(ts, pm-1.96*np.sqrt(pv), pm+1.96*np.sqrt(pv), alpha=0.3, color='blue')
		canvas.draw()


	def correct(self, ax, canvas, rbpf):
		aprev = rbpf.get_state_mean()
		cprev = rbpf.get_state_covariance()

		for particle in rbpf.particles:
			particle.correct(rbpf.current_price)
		rbpf.reweight_particles()

		if rbpf.get_logDninf() < rbpf.log_resample_limit:
				rbpf.resample_particles()

		ts = np.array([rbpf.prev_time, rbpf.current_time])
		pm = np.array([aprev[0,0], rbpf.get_state_mean()[0,0]])
		pv = np.array([cprev[0,0], rbpf.get_state_covariance()[0,0]])
		ax.plot(ts, pm, c='orange', ls='--')
		ax.fill_between(ts, pm-1.96*np.sqrt(pv), pm+1.96*np.sqrt(pv), alpha=0.3, color='blue')
		canvas.draw()


window = tk.Tk()
app = Application(sampled_data, master=window)
window.title('Particle Filter')
window.geometry("600x600")
window.mainloop()