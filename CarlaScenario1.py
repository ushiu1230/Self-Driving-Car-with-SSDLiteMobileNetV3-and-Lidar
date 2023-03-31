import carla 
import math
import random
import numpy as np 
import threading
import time
import torch
import cv2
import queue

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
IM_HEIGHT = 320
IM_WIDTH = 320 
def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (IM_HEIGHT, IM_WIDTH, 4))


#------------------------------------------------------------
# Func process frame
def get_frame(data):
    while True:
        frame = data[:,:,:3]
        frame_queue.put(frame)

# Func object detection
def object_detection():
    while True:
        frame = frame_queue.get()
        print(frame)
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
        cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('Detection', det_image)
		# press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()




def main(): 
    #----------------------------------------------------------------------------------------------------------
    # SET UP CONNECTION 
    # Connect to the client and retrieve the world object
    #client = carla.Client('192.168.1.9', 2000)
    client = carla.Client('26.146.230.217', 2000)
    world = client.get_world()

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = False # Enables synchronous mode
    #settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)

    # Set up the TM in synchronous mode
    traffic_manager = client.get_trafficmanager()
    traffic_manager.set_synchronous_mode(False)

    # Set a seed so behaviour can be repeated if necessary
    traffic_manager.set_random_device_seed(0)
    random.seed(0)

    # We will aslo set up the spectator so we can see what we do
    spectator = world.get_spectator()



    #----------------------------------------------------------------------------------------------------------
    # GET SOME SPAWN POINT ON MAP AND DEFINE THE AUTOPILOT ROUTE

    spawn_points = world.get_map().get_spawn_points()

    # Draw the spawn point locations as numbers in the map
    # for i, spawn_point in enumerate(spawn_points):
    #     world.debug.draw_string(spawn_point.location, str(i), life_time=10)

    # Route 1
    spawn_point_1 =  spawn_points[60]
    spawn_point_2 =  spawn_points[10]
    spawn_point_3 =  spawn_points[47]

    # Create route 1 from the chosen spawn points
    route_1_indices = [149, 21, 105, 52, 104, 140, 10]
    route_1 = []
    for ind in route_1_indices:
        route_1.append(spawn_points[ind].location)

    # Now let's print them in the map so we can see our routes
    world.debug.draw_string(spawn_point_1.location, 'Spawn point 1', life_time=30, color=carla.Color(255,0,0))
        
    # for ind in route_1_indices:
    #     spawn_points[ind].location
    #     world.debug.draw_string(spawn_points[ind].location, str(ind), life_time=60, color=carla.Color(255,0,0))

    #----------------------------------------------------------------------------------------------------------
    # Get blueprint
    # Vehicle
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("model3")[0]
    npc_bp = bp_lib.filter("cooper_s")[0]
    print(vehicle_bp)
    print(npc_bp)
    
    # Camera
    cam_bp = bp_lib.find("sensor.camera.rgb")
    cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
    cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
    cam_bp.set_attribute("fov", "110")


    # Spawn actor
    # Vehicle
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
    npc1 = world.try_spawn_actor(npc_bp, spawn_point_2)
    npc2 = world.try_spawn_actor(npc_bp, spawn_point_3)

    # Cam
    spawn_cam_point = carla.Transform(carla.Location(x=1, z=1.5))
    cam_sensor = world.spawn_actor(cam_bp, spawn_cam_point, attach_to=vehicle)
    camera_data = {'image': np.zeros((IM_HEIGHT, IM_WIDTH, 3))}

    # Set view on our Actor
    transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
    spectator.set_transform(transform)

    # Set route for auto pilot
    traffic_manager.set_path(vehicle, route_1)

    # Set maximum speed 
    traffic_manager.global_percentage_speed_difference(70)

    # Ignore_Lights
    traffic_manager.ignore_lights_percentage(vehicle, 100)
    
    # Setup car engine
    vehicle.apply_physics_control(carla.VehiclePhysicsControl(max_rpm = 100.0, center_of_mass = carla.Vector3D(0.0, 0.0, 0.0), torque_curve=[[0,40],[100,40]]))


    # Set Angle and Speed for car
    manual_mode = True
    auto_mode = True
    count = 300

    #----------------------------------------------------------------------------------------------------------
    cam_sensor.listen(lambda image: camera_callback(image, camera_data))
    
    get_frame_thread = threading.Thread(target=get_frame, args=(camera_data['image'],))
    get_frame_thread.start()

    object_detection_thread = threading.Thread(target=object_detection)
    object_detection_thread.start()

    # *******CONTROL******
    # In synchronous mode, we need to run the simulation to fly the spectator


    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)

    while True:
        world.tick()

        #set viewpoint at main actor
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        print(camera_data['image'])
        print(camera_data['image'][:,:,:3])
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
        # det_image = cv2.cvtColor(det_image, cv2.COLOR_BGR2RGB)
        cv2.imshow('RGB Camera', det_image)
        # press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        if (manual_mode and count != 0):
            print("tick: ", count, "speed: ", vehicle.get_acceleration())
            if (count >= 161):
                vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=0))
                count = count - 1
            elif (count >= 81 and count < 161):
                vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=-0.35))
                count = count - 1
            elif (count > 0 and count < 81):
                vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=0.35))
                count = count - 1
        else: 
            manual_mode = False
            vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
            vehicle.set_autopilot(auto_mode) # Give TM control over vehicle
    if cam_bp:
        cam_bp.detroy()
    if vehicle:
        vehicle.destroy()
    cv2.destroyAllWindows()

    get_frame_thread.join()
    object_detection_thread.join()
        
if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(' - Exited by user.')
