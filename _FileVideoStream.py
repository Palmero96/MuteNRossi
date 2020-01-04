# import the necessary packages
from imutils.video import FPS
import numpy as np
import imutils
import cv2 as cv
import threading
import queue
import time

class FileVideoStream:
	def __init__(self, path, queueSize=1):
		self.stream = cv.VideoCapture(path)
		self.stopped = False

		self.Q = queue.Queue(maxsize=queueSize)

	def start(self):
		t = threading.Thread(target=self.update, args=())
		t.daemon = True
		t.start()
		return self

	def update(self):
		while True:
			if self.stopped:
				return

			if not self.Q.full():
				grabbed,frame = self.stream.read()

				if not grabbed:
					self.stop()
					return

				self.Q.put(frame)

	def read(self):
		return self.Q.get()

	def more(self):
		return self.Q.qsize() > 0

	def stop(self):
		self.stopped = True

#############################################################


#
# while True:
# 	frame = fvs.read()
# 	frame = imutils.resize(frame, width=450)
# 	frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
# 	frame = np.dstack([frame, frame, frame])
#
# 	# display the size of the queue on the frame
# 	cv.putText(frame, "Queue Size: {}".format(fvs.Q.qsize()),
# 		(10, 30), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
#
# 	# show the frame and update the FPS counter
# 	cv.imshow("Frame", frame)
# 	k = cv.waitKey(1) & 0xFF
# 	if k==ord('q'):
# 		break
# 	fps.update()
#
# # stop the timer and display FPS information
# fps.stop()
# print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
# print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
#
# # do a bit of cleanup
# cv.destroyAllWindows()
# fvs.stop()
