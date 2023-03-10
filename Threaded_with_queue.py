# Python program to illustrate the concept of threading object detection task
# importing the threading module
import threading
import time
import torch
import cv2
import queue

from coco_names import coco_names
from model import *
from detect_utils import *

#Define queue to store frames
frame_queue = queue.Queue(maxsize=1)

#define model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Model4(device)
model = model.half()
model = model.to(device)


#func process frame
def get_frame():
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame_pos = cv2.resize(frame, (320,320))
        frame_queue.put(frame_pos)
    cap.release()

#fund object detection
def object_detection():
    while True:
        frame = frame_queue.get()
        start_time = time.time()
        boxes, classes, labels = predict(frame, model, device, 0.9)
        # get predictions for the current frame  
        # draw boxes
        det_image = draw_boxes(boxes, classes, labels, frame)
        fps = 1/(time.time() - start_time)
        # write the FPS on the current frame
        cv2.putText(det_image, f"{fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		#convert from BGR to RGB color format
        det_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('Detection', det_image)
		# press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_frame_thread = threading.Thread(target=get_frame)
    get_frame_thread.start()

    object_detection_thread = threading.Thread(target=object_detection)
    object_detection_thread.start()

    get_frame_thread.join()
    object_detection_thread.join()
