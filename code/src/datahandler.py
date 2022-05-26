import pandas as pd
import datetime as dt


class TimeseriesData:
	# def __init__(self, path):
	# 	self.path = path
	# 	self.df = pd.read_csv(self.path, sep=',', parse_dates=[['Date', 'Time']])
	# 	# times are initially in reverse order
	# 	self.df = self.df[::-1]
	# 	self.df.reset_index(inplace=True, drop=True)
	# 	self.df['Date_Time'] = self.df['Date_Time'].subtract(self.df['Date_Time'][0])
	# 	self.df['Date_Time'] = self.df['Date_Time'].dt.total_seconds()
	# 	print(self.df)


	def __init__(self, path, idx1=0):
		self.path = path
		dftemp = pd.read_csv(self.path, sep=',')
		self.df = pd.DataFrame(dftemp[['Telapsed', 'Price']][idx1:])


	def remove_non_unique(self, ret=False):
		self.df_utimes = self.df.drop_duplicates(subset='Date_Time', keep='first', ignore_index=False, inplace=False)
		self.df_utimes.reset_index(inplace=True, drop=True)
		if ret:
			return self.df_utimes


	def plot(self, ax):
		ax.plot(self.df['Telapsed'], self.df['Price'])


class TickData:
	def __init__(self, path, nrows=0):
		self.path = path
		self.df = pd.read_csv(self.path, sep=',', header=0, usecols=['DateTime', 'Bid', 'Ask'], nrows=nrows)
		self.df = self.df.drop_duplicates(subset='DateTime', keep='first', ignore_index=False)
		self.df['DateTime'] = pd.to_datetime(self.df['DateTime'], format='%Y%m%d %H:%M:%S.%f')
		self.df['DateTime'] = (self.df['DateTime'].subtract(self.df['DateTime'][0])).dt.total_seconds()


	def plot(self, ax):
		ax.plot(self.df['DateTime'], self.df['Bid'], color='black', ls='--', lw=0., marker='.', ms=2., mec='black', mfc='black')
		ax.set_xticks([])