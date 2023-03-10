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
result_queue = queue.Queue(maxsize=1)

#define model
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = Model4(device)
model = model.half()
model.to(device)


#func process frame
def get_frame():
    cap = cv2.VideoCapture(0)
    while True:
        ret,frame = cap.read()
        if not ret:
            break
        frame_queue.put(frame)
    cap.release()

#fund object detection
def object_detection(frame):
        with torch.no_grad():
            boxes, classes, labels = predict(frame, model, device, 0.9)
            # get predictions for the current frame  
        # draw boxes
        det_image = draw_boxes(boxes, classes, labels, frame)
        result_queue.put(det_image)

def display_result():
    prev_time = time.time()
    fps_counter = 0
    while True:
        frame = frame_queue.get()
        if frame is None:
            break
        # thread for object detect
        object_detection_thread = threading.Thread(target=object_detection, args=(frame,))
        object_detection_thread.start()
        object_detection_thread.join()
        result = result_queue.get()

        #display
        result_img = cv2.cvtColor(result, cv2.COLOR_RGB2BGR)
        cv2.imshow("Det", result_img)

        #show FPS
        fps_counter += 1
        curr_time = time.time()
        if curr_time - prev_time >=1:
            fps = fps_counter/(curr_time - prev_time)
            print(f'FPS:{fps:.2f}')
            prev_time = curr_time
            fps_counter = 0
        
        if cv2.waitKey(1) == ord('q'):
            break

    #release mem
    cap.release()
    cv2.destroyAllWindows()

frame_thread = threading.Thread(target=get_frame)
display_thread = threading.Thread(target=display_result)
frame_thread.start()
display_thread.start()

frame_thread.join()
display_thread.join()