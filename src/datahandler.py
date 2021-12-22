import pandas as pd


class TimeseriesData:
	def __init__(self, path):
		self.path = path
		self.df = pd.read_csv(self.path, sep=',')
		self.df = self.df[::-1]


	def remove_non_unique(self, ret=False):
		self.df_utimes = self.df.drop_duplicates(subset='Time', keep='first', ignore_index=False, inplace=False)
		if ret:
			return self.df_utimes

