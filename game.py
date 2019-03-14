import cv2
import numpy as np
import pickle
import pandas as pd
import os
import datetime
from locations import overlap_test, calculate_overlap, smallest_bounding_box

class closing(object):
	def __init__(self, game):
		self.game = game

	def __enter__(self):
		return self.game

	def __exit__(self, *exc_info):
		try:
			self.game.close()
		except AttributeError:
			print("AttributeError when closing ####")

class Game:

	def __init__(self, imagepath):
		self.window_height = -1
		self.window_width = -1
		self.WINDOW_NAME = "test"
		self.current_imagepath = imagepath
		self.current_bounding_boxes = []

		self.drawing = False # True if mouse is pressed
		ix, iy = -1, -1
		self.img_background = np.zeros((512,512,3), np.uint8)
		self.img_background_cache = self.img_background.copy()
		self.img_drawing = self.img_background.copy()
		self.label_dict = {}
		self.label_index = 1

		self.img_height_original = -1
		self.img_width_original = -1

		# Pre-load labels (answers to where parasites are)
		with open('dataframe_labels.pickle', 'rb') as pid:
			self.dataframe_labels = pickle.load(pid)
		self._get_bounding_boxes(self.current_imagepath.split("/")[-1])

		self.color_boxtype_palette = [(0,0,255), (255,255,255), (0,255,0)]

		currenttime = self.timestamp()
		logdir = 'logs/{}_{}.log'.format(self.current_imagepath.split('/')[-1].split('.')[0], currenttime)
		self.logid = open(logdir, 'w')
		print("Writing to {}".format(logdir))
		print(id(self))
		self.logid.write('datetime,x1,y1,x2,y2,flag\n')

	def timestamp(self):
		ct = datetime.datetime.now()
		currenttime = "{}{}{}{}{}{}{}".format(ct.year, ct.month, ct.day, ct.hour, ct.minute, ct.second, ct.microsecond)
		return currenttime

	def close(self):
		self.logid.close()
		print("CLOSED")

	def _get_bounding_boxes(self, imagename):
		df = pd.DataFrame()
		image_index = self.dataframe_labels.loc[self.dataframe_labels['filename'] == imagename].index.values.astype(int)[0]
		print("Labelling image {}".format(image_index))
		overlaps = calculate_overlap(self.dataframe_labels, image_index)
		bounding_boxes = []
		for overlap in overlaps:
			bounding_boxes.append(smallest_bounding_box(overlap))
		self.current_bounding_boxes = bounding_boxes

	def _format_box(self, pt1, pt2):
		"""
		Make sure first point is in upper left corner, and second in lower right
		"""
		x1 = np.min([pt1[0], pt2[0]])
		y1 = np.min([pt1[1], pt2[1]])
		x2 = np.max([pt1[0], pt2[0]])
		y2 = np.max([pt1[1], pt2[1]])

		return (x1,y1),(x2,y2)

	def _coords_to_original(self, pt1, pt2):
		"""
		Returns coordinates as would appear in the original (not rescaled) image
		"""
		(x1,y1),(x2,y2) = self._format_box(pt1,pt2)
		x1_orig = int(x1 * (self.img_width_original/self.window_width))
		y1_orig = int(y1 * (self.img_height_original/self.window_height))
		x2_orig = int(x2 * (self.img_width_original/self.window_width))
		y2_orig = int(y2 * (self.img_height_original/self.window_height))
	
		return (x1_orig, y1_orig), (x2_orig, y2_orig)

	def _coords_from_original(self, pt1, pt2):
		"""
		Returnds coordinates as would appear in rescaled image (pass original coords)
		"""
		(x1,y1),(x2,y2) = self._format_box(pt1,pt2)
		x1_resc = int(x1 * (self.window_width/self.img_width_original))
		y1_resc = int(y1 * (self.window_height/self.img_height_original))
		x2_resc = int(x2 * (self.window_width/self.img_width_original))
		y2_resc = int(y2 * (self.window_height/self.img_height_original))
	
		return (x1_resc, y1_resc), (x2_resc, y2_resc)


	def _get_overlap(self, pt1, pt2, original_coords=False):
		"""
		Use disjoint sets algorithm to check for overlap with pre calculated boxes (which should represent the "true" locations)
		"""
		# Transform to pixel coordinates in original image
		if ~original_coords:
			(x1,y1),(x2,y2) = self._format_box(pt1,pt2)
			(x1_orig, y1_orig), (x2_orig, y2_orig) = self._coords_to_original((x1,y1), (x2, y2))
		#x1_orig = int(x1 * (self.img_width_original/self.window_width))
		#y1_orig = int(y1 * (self.img_height_original/self.window_height))
		#x2_orig = int(x2 * (self.img_width_original/self.window_width))
		#y2_orig = int(y2 * (self.img_height_original/self.window_height))
		#TODO Fix formatting of dictionaries to use Filip's format
		label_original = {}
		label_original['x1'] = str(x1_orig)
		label_original['y1'] = str(y1_orig)
		label_original['x2'] = str(x2_orig)
		label_original['y2'] = str(y2_orig)

		for box in self.current_bounding_boxes:
			if overlap_test([label_original], [box]):
				return box
		return None

	def _add_log_label(self, label):
		"""
		Helper function to manage adding labels to the log.
		"""
		x1 = label[0][0]
		y1 = label[0][1]
		x2 = label[1][0]
		y2 = label[1][1]
		flag = label[2]
		print(label)
		currenttime = self.timestamp()
		output = "{},{},{},{},{},{}\n".format(currenttime,x1,y1,x2,y2,flag)
		self.logid.write(output)
		self.logid.flush()

	def _mouse_callback(self, event, x, y, flags, param):
		"""
		INPUT: Standard inputs for mouse callback function in opencv
		"""
		global ix, iy, drawing

		if event == cv2.EVENT_LBUTTONDOWN:
			self.drawing = True
			ix, iy = x, y

		elif event == cv2.EVENT_MOUSEMOVE:
			if self.drawing == True:
				cv2.rectangle(self.img_drawing, (ix, iy), (x, y), (0,255,0), 2)
				self.img_background_cache = self.img_drawing.copy()
				self.img_drawing = self.img_background.copy()

		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			#boxtype = 0 # 0:wrong, 1:correct, 2:answer box
			correct=False
			labelname = "label_{:02d}".format(self.label_index)

			overlapped_box = self._get_overlap((ix,iy),(x,y))
			if overlapped_box is not None:
				ix_ans = int(overlapped_box["x1"])
				iy_ans = int(overlapped_box["y1"])
				x_ans = int(overlapped_box["x2"])
				y_ans = int(overlapped_box["y2"])
				pt1, pt2 = self._coords_from_original((ix_ans, iy_ans),(x_ans, y_ans))
				self._add_label("{}_ans".format(labelname), self.label_dict, pt1, pt2, boxtype=2, original_coords=True)
				correct = True
				print(overlapped_box)

			print(correct)
			if correct:
				self._add_label(labelname, self.label_dict, (ix,iy), (x,y), boxtype=1)
			else:
				self._add_label(labelname, self.label_dict, (ix,iy), (x,y), boxtype=0)
			self.label_index += 1
			print(self.label_dict)
			self._draw_labels()

	def _add_label(self, labelname, label_dict, firstpoint, secondpoint, boxtype=-1, original_coords=False):
		"""
		This function writes the label in a correct way to pass to other parts
		of the script. Otherwise ix,iy will not always represent the upper left
		corner and the other script won't work.

		It also adds a log entry that a label has been added. Labels are added after a mouse click.
		"""
		ix,iy = firstpoint
		x,y = secondpoint
		upper_left = (np.min([ix,x]), np.min([iy,y]))
		lower_right = (np.max([ix,x]), np.max([iy,y]))
		self.label_dict[labelname] = [upper_left, lower_right, boxtype]
		
		self._add_log_label(self.label_dict[labelname])
		
	def _draw_labels(self):
		"""
		Iterate through stored labels and put them on image
		"""
		for label_name in self.label_dict.keys():
			(x1, y1), (x2, y2), boxtype = self.label_dict[label_name]
			color_tuple = self.color_boxtype_palette[boxtype]
			if boxtype == 0 or boxtype == 2:
				cv2.rectangle(self.img_background, (x1,y1), (x2,y2), color_tuple, 2)
			elif boxtype == -1:
				print("Error: uninitialised box?")

	def run(self):
		self.img_original = cv2.imread(self.current_imagepath)
		(self.img_height_original, self.img_width_original, _) = self.img_original.shape
		# Define window size by original image dim // scaling factor
		self.window_height = self.img_height_original // 3
		self.window_width = self.img_width_original // 3

		self.img_background = cv2.resize(self.img_original, (self.window_width, self.window_height))
		cv2.namedWindow('image')
		cv2.setMouseCallback('image', self._mouse_callback)
		

		while(True):
			while(self.drawing):
				#TODO Fix green flicker
				cv2.imshow('image', self.img_background_cache)
				k = cv2.waitKey(1) & 0xFF
				if k == 27:
					break
			cv2.imshow('image', self.img_background)	
			k = cv2.waitKey(1) & 0xFF
			if k == 27:
				break
	
		cv2.destroyAllWindows()
	
for i in range(0, 10):
	images_dir = "./images_game"	
	images = os.listdir(images_dir)
	imagepath = "{}/{}".format(images_dir, images[i])
	with closing(Game(imagepath)) as game:
		game.run()
