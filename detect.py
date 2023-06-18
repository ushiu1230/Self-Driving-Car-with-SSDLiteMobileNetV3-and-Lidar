from utils.coco_names import coco_names
from models.Model import *
from utils.detect_utils import *
import time
import torch
import cv2

# construct the argument parser

# define the computation device
# select device (whether GPU or CPU)
device = torch.device('cuda')
model = Model1(device)

cap = cv2.VideoCapture(0)
if (cap.isOpened() == False):
    print('Error while trying to read video. Please check path again')
# get the frame width and height
frame_count = 0 # to count total frames
total_fps = 0 # to get the final frames per second

# read until end of video
while(cap.isOpened()):
    # capture each frame of the video
    ret, frame = cap.read()
    if ret == True:
        # get the start time
        start_time = time.time()
        with torch.no_grad():
            boxes, classes, labels = predict(frame, model, device, 0.9)
            # get predictions for the current frame  
        # draw boxes and show current frame on screen
        image = draw_boxes(boxes, classes, labels, frame)
        # get the end time
        end_time = time.time()
        # get the fps
        fps = 1 / (end_time - start_time)
        # add fps to total fps
        total_fps += fps
        # increment frame count
        frame_count += 1
        # write the FPS on the current frame
        cv2.putText(image, f"{fps:.3f} FPS", (15, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)
        # convert from BGR to RGB color format
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        cv2.imshow('image', image)
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
# release VideoCapture()
cap.release()
# close all frames and video windows
cv2.destroyAllWindows()
# calculate and print the average FPS
avg_fps = total_fps / frame_count
print(f"Average FPS: {avg_fps:.3f}")