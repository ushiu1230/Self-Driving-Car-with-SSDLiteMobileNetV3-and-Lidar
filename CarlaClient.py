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

#write log file location
import logging 
logging.basicConfig(filename="location.log", format='%(asctime)s %(message)s', filemode= 'w')
logger = logging.getLogger()
logger.setLevel(logging.DEBUG)


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
npc_bp = bp_lib.filter("cooper_s")[0]
print(vehicle_bp)

#init npc and actor spawn location
spawn_points = world.get_map().get_spawn_points()
#npc_1_spawn_point = carla.Transform(carla.Location(x=-33.574535, y=130.005219, z=0.001447), carla.Rotation(pitch=0.068548, yaw=-179.642746, roll=-0.000061))
#npc_2_spawn_point = carla.Transform(carla.Location(-18.745386, 130.095306, 0.001928), carla.Rotation(-0.014343, -179.649887, 0.000042))
for i, spawn_point in enumerate(spawn_points):
    world.debug.draw_string(spawn_point.location, str(i), life_time=10)
    logger.info(spawn_point.location)
while True:
    world.tick()


exit()
spectator = world.get_spectator()
transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
#vehicle.set_autopilot(True)
spectator.set_transform(transform)
#exit()

# #camera setup
# cam_bp = bp_lib.find("sensor.camera.rgb")
# cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
# cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
# cam_bp.set_attribute("fov", "110")
# spawn_cam_point = carla.Transform(carla.Location(x=1, z=1.5))
# cam_sensor = world.spawn_actor(cam_bp, spawn_cam_point, attach_to=vehicle)
# camera_data = {'image': np.zeros((IM_HEIGHT, IM_WIDTH, 3))}
# cam_sensor.listen(lambda image: camera_callback(image, camera_data))

#vehicle.set_autopilot(True)
# i = 0
# while i < 20:
#     time.sleep(4)
#     logger.info(vehicle.get_transform())
#     i+=1

# while True:
#     frame = camera_data['image'][:,:,:3]
#     start_time = time.time()
#     boxes, classes, labels = predict(frame, model, device, 0.9)
#     # get predictions for the current frame  
#     # draw boxes
#     det_image = draw_boxes(boxes, classes, labels, frame)
#     fps = 1/(time.time() - start_time)
#     # write the FPS on the current frame
#     cv2.putText(det_image, f"{fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
#     #convert from BGR to RGB color format
#     det_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
#     cv2.imshow('Detection', det_image)
#     # press `q` to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# cv2.destroyAllWindows()