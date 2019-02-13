import cv2
import numpy as np

class Game:

	def __init__(self):
		self.window_height = 600
		self.window_width = 640
		self.WINDOW_NAME = "test"

		self.drawing = False # True if mouse is pressed
		ix, iy = -1, -1
		self.img_background = np.zeros((512,512,3), np.uint8)
		self.img_background_cache = self.img_background.copy()
		self.img_drawing = self.img_background.copy()
		self.label_dict = {}
		self.label_index = 1

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
				cv2.rectangle(self.img_drawing, (ix, iy), (x, y), (0,255,0), 3)
				self.img_background_cache = self.img_drawing.copy()
				self.img_drawing = self.img_background.copy()

		elif event == cv2.EVENT_LBUTTONUP:
			self.drawing = False
			self.label_dict["label_{}".format(str(self.label_index))] = [(ix, iy), (x, y)]
			self.label_index += 1
			self._draw_labels()
		
	def _draw_labels(self):
		"""
		Iterate through stored labels and put them on image
		"""
		for label_name in self.label_dict.keys():
			(x1, y1), (x2, y2) = self.label_dict[label_name]
			cv2.rectangle(self.img_background, (x1,y1), (x2,y2), (255,255,255), 3)

	def run(self, imagepath):
#		im = cv2.imread(imagepath)                        # Read image
#		imS = cv2.resize(im, (self.window_width, self.window_height))                    # Resize image
#		cv2.imshow("output", imS)                            # Show image
#		cv2.waitKey(0)
		self.img = np.zeros((512,512,3), np.uint8)
		cv2.namedWindow('image')
		cv2.setMouseCallback('image', self._mouse_callback)

		while(True):
			while(self.drawing):
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
game.run("./image_001.jpg")
