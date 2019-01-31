import numpy as np
import pandas as pd
import json
import copy

"""
DATA: "labels" column in dataframe contains a list of json disctionaries for every image. Each json dictionary contains information for an individual box (=parasite).

SERVER: Run and maintained by Filip Ayazi (fa344)
"""

url_server = "https://filip.ayazi.sk/labeller/api.php?key=b7ca50e53329c&data&fbclid=IwAR3UNpOwXw3BaePHLMrsrKhKONRbx8AdbH7GOmYhEjA1P42jYqV3k44OUF0"

def load_labels(url=url_server, filename=None):
	"""
	INPUT: url to download json from, or filename
	OUTPUT: Dataframe read from json
	"""
	df = pd.DataFrame()
	if filename is None:
		try:
			df = pd.read_json(url)
			print("Read data from server")
		except:
			print("ERROR: Failed reading from server")
	else:
		try:
			df = pd.read_json(filename)
			print("Read data from {}".format(filename))
		except:
			print("Failed reading from {}".format(filename))

	return df

def select_from_dataframe(df, select_rows=["filename", "labels"]):
	"""
	INPUT: Dataframe to be worked on, rows to selecto
	OUTPUT: Dataframe modified to contain only selected rows
	"""
	df_new = pd.DataFrame()
	for col in select_rows:	
		df_new[col] = df[col]

	return df_new

def people_to_cols(df):
	names = []
	print("Found data from:\n")
	for label in df["labels"].iloc[0]:
		if label["username"] not in names:
			names.append(label["username"])
			print(label["username"])

	labels_from_name_init = {}
	for name in names:
		df[name] = np.nan
		labels_from_name_init[name] = []

	labels_from_name = copy.deepcopy(labels_from_name_init.copy())
	for i in range(0, df.index.size):
		for json in df["labels"].iloc[i]:
			for name in names:
				if json["username"] == name:
					labels_from_name[name].append(json)
		for name in names:
			df[name].iloc[i] = labels_from_name[name]
		labels_from_name = copy.deepcopy(labels_from_name_init)

	df.drop(["labels"], axis=1, inplace=True)
	return df

	
def total_number_parasites(df):
	names = list(df.columns)
	names = names[1:len(names)]
	
	for name in names:
		df[name + "number"] = np.nan

	for i in range(0, df.index.size):
		for name in names:
			df[name + "number"].iloc[i] = len(df[name].iloc[i])

	for name in names:
		df[name + "number"] = df[name + "number"].astype('uint8')
			
	return df



df = select_from_dataframe(load_labels())
df = people_to_cols(df)
df = total_number_parasites(df)
print(df)







