import cv2
import numpy as np
import pickle
import pandas as pd
from locations import overlap_test, calculate_overlap, smallest_bounding_box

class Game:

	def __init__(self):
		self.window_height = 600
		self.window_width = 640
		self.WINDOW_NAME = "test"
		self.current_imagepath = './image_001.jpg'
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

	def _get_bounding_boxes(self, imagename):
		df = pd.DataFrame()
		image_index = self.dataframe_labels.loc[self.dataframe_labels['filename'] == imagename].index.values.astype(int)[0]
		print("Labelling image {}".format(image_index))
		overlaps = calculate_overlap(self.dataframe_labels, image_index)
		bounding_boxes = []
		for overlap in overlaps:
			bounding_boxes.append(smallest_bounding_box(overlap))
		self.current_bounding_boxes = bounding_boxes

	def _check_correct(self, label_name):
		"""
		Use disjoint sets algorithm to check for overlap with pre calculated boxes (which should represent the "true" locations)
		"""
		# Transform to pixel coordinats in original image
		(x1,y1), (x2,y2), _ = self.label_dict[label_name]
		x1_orig = int(x1 * (self.img_width_original/self.window_width))
		y1_orig = int(y1 * (self.img_height_original/self.window_height))
		x2_orig = int(x2 * (self.img_width_original/self.window_width))
		y2_orig = int(y2 * (self.img_height_original/self.window_height))
		#TODO Fix formatting of dictionaries to use Filip's format
		label_original = {}
		label_original['x1'] = str(x1_orig)
		label_original['y1'] = str(y1_orig)
		label_original['x2'] = str(x2_orig)
		label_original['y2'] = str(y2_orig)

		return overlap_test([label_original], self.current_bounding_boxes)

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
				cv2.rectangle(self.img_drawing, (ix, iy), (x, y), (0,255,0), 1)
				self.img_background_cache = self.img_drawing.copy()
				self.img_drawing = self.img_background.copy()

		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			correct = False
			labelname = "label_{:02d}".format(self.label_index)
			self.label_dict[labelname] = [(ix, iy), (x, y), correct]
			correct = self._check_correct(labelname)
			self.label_dict[labelname] = [(ix, iy), (x, y), correct]
			print(correct)
			self.label_index += 1
			self._draw_labels()
		
	def _draw_labels(self):
		"""
		Iterate through stored labels and put them on image
		"""
		for label_name in self.label_dict.keys():
			(x1, y1), (x2, y2), _ = self.label_dict[label_name]
			cv2.rectangle(self.img_background, (x1,y1), (x2,y2), (255,255,255), 1)

	def run(self):
		self.current_imagepath = self.current_imagepath
		self.img_original = cv2.imread(self.current_imagepath)
		(self.img_height_original, self.img_width_original, _) = self.img_original.shape

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
		
	
game = Game()
game.run()
