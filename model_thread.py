# Python program to illustrate the concept
# of threading
# importing the threading module
import threading
import time
from coco_names import coco_names
from model import *
from detect_utils import *
import time
import torch
import cv2
# construct the argument parser

# define the computation device
# select device (whether GPU or CPU)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print()
model = Model1(device)
model = model.half()
model.to(device)

def videostream(lock):
	global frame
	
	while(1):
		lock.acquire()
		ret, frame = cap.read()
		lock.release()

def detecting(lock):
	global frame
	global det_image
	global start_time
	while(1):
		lock.acquire()
		frame_count = 0 # to count total frames
		total_fps = 0 # to get the final frames per second
		start_time = time.time()
		with torch.no_grad():
			boxes, classes, labels = predict(frame, model, device, 0.9)
			# get predictions for the current frame  
		# draw boxes and show current frame on screen
		det_image = draw_boxes(boxes, classes, labels, frame)

		print("dang chay")
		end_time = time.time()
		# get the fps
		fps = 1 / (end_time - start_time)
		# add fps to total fps
		total_fps += fps
		# increment frame count
		frame_count += 1
		# write the FPS on the current frame
		cv2.putText(det_image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
					1, (0, 255, 0), 2)
		#convert from BGR to RGB color format
		det_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
		cv2.imshow('Detection', det_image)
		# press `q` to exit
		if cv2.waitKey(1) & 0xFF == ord('q'):
			break

		lock.release()

if __name__ =="__main__":
	# # creating thread
	# t1 = threading.Thread(target= thread_1)
	# t2 = threading.Thread(target= thread_2)

	# # starting thread 1
	# t1.start()
	# # starting thread 2
	# t2.start()

	# # wait until thread 1 is completely executed
	# t1.join()
	# # wait until thread 2 is completely executed
	# t2.join()

	# # both threads completely executed
	# print("Done!")
	total_fps = 0
	frame_count = 0
	start_time = 0 
	cap = cv2.VideoCapture(0)
	if (cap.isOpened() == False):
		print('Error while trying to read video. Please check path again')
	ret, frame = cap.read()
	lock = threading.Lock()
	t1 = threading.Thread(target=videostream, args= (lock,))
	t2 = threading.Thread(target=detecting, args= (lock,))
	# starting thread 1
	t1.start()
	# starting thread 2
	t2.start()

	# wait until thread 1 is completely executed
	t1.join()
	# wait until thread 2 is completely executed
	t2.join()

	# release VideoCapture()
	cap.release()
	# close all frames and video windows
	cv2.destroyAllWindows()
	avg_fps = total_fps / frame_count
	print(f"Average FPS: {avg_fps:.3f}")