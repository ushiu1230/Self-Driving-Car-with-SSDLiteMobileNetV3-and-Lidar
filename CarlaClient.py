import carla 
import math
import random
import numpy as np 
import threading
import time
import torch
import cv2
import queue
import carla

from utils.coco_names import coco_names
from models.Model4 import *
from utils.detect_utils import *

#------------------------------------------------------------
#Define queue to store frames
frame_queue = queue.Queue(maxsize=1)
#------------------------------------------------------------
#define model
device = torch.device('cuda')
model = Model4(device)
model = model.half()
model = model.to(device)
#------------------------------------------------------------

def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (IM_HEIGHT, IM_WIDTH, 4))
#------------------------------------------------------------

IM_HEIGHT = 320
IM_WIDTH = 320 

client = carla.Client('192.168.1.9', 2000)

world = client.get_world()

bp_lib = world.get_blueprint_library()
vehicle_bp = bp_lib.filter("model3")[0]
print(vehicle_bp)

#init npc and actor spawn location
spawn_points = random.choice(world.get_map().get_spawn_points())
vehicle = world.try_spawn_actor(vehicle_bp, spawn_points)
spectator = world.get_spectator()
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
spectator.set_transform(transform)

vehicle.set_autopilot(True)

#camera setup
cam_bp = bp_lib.find("sensor.camera.rgb")
cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
cam_bp.set_attribute("fov", "110")
spawn_cam_point = carla.Transform(carla.Location(x=1, z=1.5))
cam_sensor = world.spawn_actor(cam_bp, spawn_cam_point, attach_to=vehicle)
camera_data = {'image': np.zeros((IM_HEIGHT, IM_WIDTH, 3))}
cam_sensor.listen(lambda image: camera_callback(image, camera_data))

# cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
# cv2.imshow('rgb camera', camera_data["image"])
# cv2.waitKey(1)

# while True:
#     cv2.imshow('rgb camera', camera_data["image"])

#     if cv2.waitKey(1) == ord('q'):
#         break

# cv2.destroyAllWindows()

while True:
    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
    frame = camera_data['image'][:,:,:3]
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
cv2.destroyAllWindows()