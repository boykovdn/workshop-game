import numpy as np
import pandas as pd
import json
import copy
import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib
import os

"""
DATA: "labels" column in dataframe contains a list of json disctionaries for every image. Each json dictionary contains information for an individual box (=parasite).

SERVER: Run and maintained by Filip Ayazi (fa344)
"""

url_server = "https://filip.ayazi.sk/labeller/api.php?key=b7ca50e53329c&data&fbclid=IwAR3UNpOwXw3BaePHLMrsrKhKONRbx8AdbH7GOmYhEjA1P42jYqV3k44OUF0"
images_dir = "./images"
images_dir_labels = "./images_labels"
colourmap = "tab10"
colours = {}
dataframe_pickle_people_name = "dataframe_people.pickle"
dataframe_pickle_labels_name = "dataframe_labels.pickle"
images_location_interest = "./images_location_interest"

# Create colour palette
cmap = plt.get_cmap(colourmap).colors
print("Loading rectangle colour palette:")
for i in range(0, len(cmap)):
	colours[str(i)]	= list(map(lambda a : int(a * 255), cmap[i]))
	print("Colour {} {}".format(i,colours[str(i)]))


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
	INPUT: Dataframe to be worked on, rows to select
	OUTPUT: Dataframe modified to contain only selected rows
	"""
	df_new = pd.DataFrame()
	for col in select_rows:	
		df_new[col] = df[col]

	return df_new

def people_to_cols(df):
	names = []
	print("Found data from:")
	for label in df["labels"].iloc[0]:
		if label["username"] not in names:
			names.append(label["username"])
			print(label["username"])

	print("\n")

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

def label_image(df, loc, labels=[], confidence=False):
	"""
	INPUT: index location of image in dataframe
	OUTPUT: image with boxes drawn in, if lebels=True
	"""
	imagename = df["filename"].iloc[loc]
	img = cv2.imread(images_dir + "/" + imagename)
	names = list(df.columns)
	names = names[1:len(names)]
	names = [name for name in names if "number" not in name]

	if labels == []:
		print("Labelling individual people's boxes with different colours")
		for i in range(0, len(names)):
			name = names[i]
			# Draw locations:
			for json in df[name].iloc[loc]:
				x1 = int(json["x1"])
				y1 = int(json["y1"])
				x2 = int(json["x2"])
				y2 = int(json["y2"])
				cv2.rectangle(img,(x1,y1), (x2,y2), colours[str(i)], 3)
	else:
		print("Labelling image from passed labels, likely single areas of interest...")
		if confidence:
			names = []
			font = cv2.FONT_HERSHEY_SIMPLEX
			fontScale = 1
			lineType = 2
			for json in df["labels"].iloc[loc]:
				for bounding_box_name in json["username"].split(","):
					if bounding_box_name not in names:
						print(names)
						names.append(bounding_box_name)

		for json in labels:
			x1 = int(json["x1"])
			y1 = int(json["y1"])
			x2 = int(json["x2"])
			y2 = int(json["y2"])
			cv2.rectangle(img,(x1,y1), (x2,y2), colours[str(1)], 3)
			if confidence:
				bottomLeftCornerOfText = (x1, y1 - 10)
				fontColor = colours[str(1)]
				print("WARNING: Confidence for each location still not working properly! You will get weird numbers for bounding boxes.") # TODO
				text_confidence = "{}/{}".format(len(json["username"].split(",")), len(names))
				cv2.putText(img, text_confidence,
							bottomLeftCornerOfText,
							font,
							fontScale,
							fontColor,
							lineType)
	
	return img
		
	
def save_images_labelled(df, target_dir=images_dir_labels):
	""" INPUT: Dataframe of images and boxes (Final version, after formatting)
	OUTPUT: None, write images with drawn boxes into target_dir. Overwrites previous contents. If target_dir does not exist, creates it.
	"""

	if os.path.exists(target_dir):
		if os.listdir(target_dir) == []:
			number_of_images = df.index.size
			print(target_dir + " empty, nothing will be overwritten.")
			for loc in range(0, number_of_images):
				cv2.imwrite(target_dir + "/" + df["filename"].iloc[loc], label_image(df, loc))
				print("[" + str(loc) + "/" + str(number_of_images-1)  + "] Copy " + df['filename'].iloc[loc] + " to " + target_dir)
		else:
			number_of_images = df.index.size	
			print(target_dir + " non-empty, overwriting conflicts")
			for loc in range(0, number_of_images):
				cv2.imwrite(target_dir + "/" + df["filename"].iloc[loc], label_image(df, loc))
				print("[" + str(loc) + "/" + str(number_of_images-1)  + "] Copy " + df['filename'].iloc[loc] + " to " + target_dir)

def show_image(img):
	"""
	Displays image
	"""
	cv2.namedWindow("test", cv2.WINDOW_NORMAL)
	cv2.imshow("test", img)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

def overlap_test(label_list1, label_list2):
	"""
	INPUT: Two sets of labels in JSON format
	OUTPUT: True if there is any overlap between any pair of labels between the two sets, False if no overlap.
	"""
	for label1 in label_list1:
		for label2 in label_list2:
			if (int(label1["y2"]) >= int(label2["y1"])) and (int(label1["x1"]) <= int(label2["x2"])) and (int(label1["x2"]) >= int(label2["x1"])) and (int(label1["y1"]) <= int(label2["y2"])):
				return True
	return False

def calculate_overlap(df, image_number):
	"""
	INPUT: Dataframe, formatted, that contains information about labels, use dataframe_pickle_labels_name; index location of image for which the locations of interest will be found
	OUTPUT: Dataframe with new rows that represent all the unique locations that could contain a parasite, and how many people have labelled them.
	"""
	#TODO Return a proper dataframe
	# Should be close to Kruskal's algorithm, but unweighted graph
	final_lists = []
	label_lists = [] # Disjoint set
	labels_all = df["labels"].iloc[image_number]
	for i in range(0, len(labels_all)):
		label_lists.append([labels_all[i]])
	
	while label_lists != []:	
		counter = 1
		while counter < len(label_lists):
			if overlap_test(label_lists[0], label_lists[counter]):
				label_lists[0] = label_lists[0] + label_lists[counter]
				label_lists.pop(counter)
			else:
				counter += 1 # Do not increment counter if element popped, otherwise will miss one! After pop everything moves an index down.

		final_lists.append(label_lists.pop(0))
		counter = 1
	return final_lists


def update_pickled_dataframe(outname):
	"""
	Updates the local pickle of the dataframe, you can choose which one - there can be multiple that have different formats, and used for different parts of the script
	"""
	if outname == dataframe_pickle_people_name:
		df = select_from_dataframe(load_labels())
		df = people_to_cols(df)
		df = total_number_parasites(df) # Final version of dataframe

	elif outname == dataframe_pickle_labels_name:
		df = select_from_dataframe(load_labels())
	
	else:
		print("ERROR: Did not recognise pickle name, did nothing!")
		
	pickle_out = open(outname,"wb")
	pickle.dump(df, pickle_out)
	pickle_out.close()
	print("Updated pickle " + outname)

def load_pickle(inname):
	"""
	Load dataframe that is locally pickled
	"""
	if inname not in [dataframe_pickle_people_name,
					  dataframe_pickle_labels_name]:
		raise Exception("{} not in standard dataframe names".format(inname))

	pickle_in = open(inname, "rb")
	df = pickle.load(pickle_in)
	pickle_in.close()
	print("Loaded {}".format(inname))

	return df

def smallest_bounding_box(labels):
	"""
	INPUT: Set of labels (rectangles) in json format
	OUTPUT: json in same format, containing xy coordinates of the smallest rectangle that bounds all of the boxes
	"""
	x2_biggest = int(labels[0]["x2"])
	x1_smallest = int(labels[0]["x1"])
	y2_biggest = int(labels[0]["y2"])
	y1_smallest = int(labels[0]["y1"])
	for json in labels:
		if int(json["x1"]) < x1_smallest:
			x1_smallest = int(json["x1"])
		if int(json["y1"]) < y1_smallest:
			y1_smallest = int(json["y1"])
		if int(json["x2"]) > x2_biggest:
			x2_biggest = int(json["x2"])
		if int(json["y2"]) > y2_biggest:
			y2_biggest = int(json["y2"])

	box = labels[0]
	for i in range(1,len(labels)):
		for tag in labels[i].keys():
			box[tag] = box[tag] + "," + labels[i][tag]

	box["x1"] = str(x1_smallest)
	box["y1"] = str(y1_smallest)
	box["x2"] = str(x2_biggest)
	box["y2"] = str(y2_biggest)
		
	return box

def save_images_loi(df):
	"""
	INPUT: Dataframe with label information (labels)
	OUTPUT: void, save locations of interest to local folder
	"""

	print("Saving bounding boxes as images in local dir...")
	with open("./{}/locations.json".format(images_location_interest), "w+") as json_output:
		for i in range(0, df.index.size):
			counter = 0
			overlaps = calculate_overlap(df, i)
			bounding_boxes = []
			for overlap in overlaps:
				bounding_box = smallest_bounding_box(overlap)
				x1 = int(bounding_box["x1"])
				y1 = int(bounding_box["y1"])
				x2 = int(bounding_box["x2"])
				y2 = int(bounding_box["y2"])
				img = cv2.imread("./{}/{}".format(images_dir, df["filename"].iloc[i]))
				filename = "./{}/{}_{}.jpg".format(images_location_interest, df["filename"].iloc[i].split(".")[0], str(counter))
				cv2.imwrite(filename, img[y1:y2,x1:x2])
				print("Saved {}".format(filename))
				json_output.write(json.dumps(bounding_box))
				counter += 1
		
			

# Run to update local pickled dataframes from server
#update_pickled_dataframe(dataframe_pickle_people_name)
#update_pickled_dataframe(dataframe_pickle_labels_name)

# Load from local - quicker
df = load_pickle(dataframe_pickle_labels_name)
#save_images_loi(df)
print(df)
#print(df)
overlaps = calculate_overlap(df, 0)
#for overlap in overlaps:
#	show_image(label_image(df, 0, labels=overlap))
big_boxes = []
for overlap in overlaps:
	big_boxes.append(smallest_bounding_box(overlap))
show_image(label_image(df, 0, labels=big_boxes))
#print(len(calculate_overlap(df, 0)))
#save_images_labelled(df)
#
#show_image(label_image(df, 0))
