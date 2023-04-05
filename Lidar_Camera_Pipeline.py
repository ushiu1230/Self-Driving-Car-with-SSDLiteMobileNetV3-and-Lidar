import carla 
import math
import random
import numpy as np 
import threading
import time
import cv2
import queue
import open3d as o3d
import torch
import keyboard
import sys

from matplotlib import cm
from utils.coco_names import coco_names
from models.Model4 import *
from utils.detect_utils import *

GLOBAL_FPS=60 # world tick rate
SENSOR_FPS=15 # sensor tick rate


#------------------------------------------------------------
# Define queue to store frames
frame_queue = queue.Queue(maxsize=1)

#------------------------------------------------------------
# Define model
device = torch.device('cuda')
model = Model4(device)
model = model.half()
model = model.to(device)

#------------------------------------------------------------
# Image size
IM_HEIGHT = 320
IM_WIDTH = 320 

#------------------------------------------------------------
# Auxilliary code for colormaps and axes
VIRIDIS = np.array(cm.get_cmap('plasma').colors)
VID_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])

COOL_RANGE = np.linspace(0.0, 1.0, VIRIDIS.shape[0])
COOL = np.array(cm.get_cmap('winter')(COOL_RANGE))
COOL = COOL[:,:3]

def add_open3d_axis(vis):
    """Add a small 3D axis on Open3D Visualizer"""
    axis = o3d.geometry.LineSet()
    axis.points = o3d.utility.Vector3dVector(np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    axis.lines = o3d.utility.Vector2iVector(np.array([
        [0, 1],
        [0, 2],
        [0, 3]]))
    axis.colors = o3d.utility.Vector3dVector(np.array([
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]]))
    vis.add_geometry(axis)

#-------------------------------------------------------------------
# Get data from camera
def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (IM_HEIGHT, IM_WIDTH, 4))

#-------------------------------------------------------------------
# Get data from Lidar
def lidar_callback(point_cloud, point_list):
    """Prepares a point cloud with intensity
    colors ready to be consumed by Open3D"""
    data = np.copy(np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4')))
    data = np.reshape(data, (int(data.shape[0] / 4), 4))

    # Isolate the intensity and compute a color for it
    intensity = data[:, -1]
    intensity_col = 1.0 - np.log(intensity) / np.log(np.exp(-0.004 * 100))
    int_color = np.c_[
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 0]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 1]),
        np.interp(intensity_col, VID_RANGE, VIRIDIS[:, 2])]

    # Isolate the 3D data
    points = data[:, :-1]
    # We're negating the y to correclty visualize a world that matches
    # what we see in Unreal since Open3D uses a right-handed coordinate system
    points[:, :1] = -points[:, :1]
    Sensor_loc = carla.Location(0,0,2.5)
    delta_x = np.power(points[:, 0] - Sensor_loc.x, 2)
    delta_y = np.power(points[:, 1] - Sensor_loc.y, 2)
    delta_z = np.power(points[:, 2] - Sensor_loc.z, 2)

    distance = np.sqrt(delta_x + delta_y + delta_z)
    print("Distance to around object: ", min(distance))

    # # An example of converting points from sensor to vehicle space if we had
    # # a carla.Transform variable named "tran":
    # points = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    # points = np.dot(tran.get_matrix(), points.T).T
    # points = points[:, :-1]

    point_list.points = o3d.utility.Vector3dVector(points)
    point_list.colors = o3d.utility.Vector3dVector(int_color)


#-------------------------------------------------------------
# Func object detection
def object_detection():
    # Start sensor
    cam_sensor.listen(lambda image: camera_callback(image, camera_data))

    # OpenCV named window for rendering
    cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RGB Camera', camera_data['image'][:, :, :3])
    cv2.waitKey(1)
    while True:
        world.tick()
        # cv2.namedWindow('RGB Camera', cv2.WINDOW_AUTOSIZE)
        # cv2.imshow('RGB Camera', camera_data['image'][:, :, :3])
        frame = camera_data['image'][:, :, :3]
        start_time = time.time()
        boxes, classes, labels = predict(frame, model, device, 0.9)
        # # get predictions for the current frame  
        # # draw boxes
        frame = draw_boxes(boxes, classes, labels, frame)
        fps = 1/(time.time() - start_time)
        # write the FPS on the current frame
        cv2.putText(frame, f"{fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
		#convert from BGR to RGB color format
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow('RGB Camera', frame)
		# press `q` to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    #--------------------------------------------------------------------------
    # Close displayws and stop sensors
    cv2.destroyAllWindows()
    sys.exit()

def lidar_pointcloud():
    # Start sensor
    point_list = o3d.geometry.PointCloud()
    lidar_sensor.listen(lambda data: lidar_callback(data, point_list))
    # Open3D visualiser for LIDAR
    vis = o3d.visualization.Visualizer()
    vis.create_window(
        window_name='Carla Lidar',
        width=480,
        height=270,
        left=240,
        top=135)
    vis.get_render_option().background_color = [0.05, 0.05, 0.05]
    vis.get_render_option().point_size = 1
    vis.get_render_option().show_coordinate_frame = True
    add_open3d_axis(vis)
    frame = 0

    while True:
        if frame == 2:
            vis.add_geometry(point_list)
        vis.update_geometry(point_list)

        vis.poll_events()
        vis.update_renderer()
        # # This can fix Open3D jittering issues:
        time.sleep(0.005)
        world.tick()
        frame += 1
        if keyboard.is_pressed("q"):
            vis.destroy_window()
            sys.exit()
if __name__ == "__main__":
        
        #----------------------------------------------------------------------------------------------------------
        # SET UP CONNECTION 
        # Connect to the client and retrieve the world object
        client = carla.Client('192.168.1.9', 2000)
        #client = carla.Client('26.146.230.217', 2000)
        client.set_timeout(2.0)
        world = client.get_world()

        # Set up the simulator in synchronous mode
        settings = world.get_settings()
        settings.synchronous_mode = True # Enables synchronous mode
        settings.fixed_delta_seconds = 1 / GLOBAL_FPS
        world.apply_settings(settings)

        # Set up the TM in synchronous mode
        traffic_manager = client.get_trafficmanager()
        # traffic_manager.set_synchronous_mode(True)

        # Set a seed so behaviour can be repeated if necessary
        traffic_manager.set_random_device_seed(0)
        random.seed(0)

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
        cam_bp = bp_lib.find("sensor.camera.rgb")
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')

        print(vehicle_bp)
        print(npc_bp)

        # Set Camera Attribute
        cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
        cam_bp.set_attribute("sensor_tick", str(1.0 / SENSOR_FPS))
        camera_data = {'image': np.zeros((IM_HEIGHT, IM_WIDTH, 4))}

        # Set Lidar Attribute
        lidar_bp.set_attribute('range', '30.0')
        lidar_bp.set_attribute('upper_fov', '0.0')
        lidar_bp.set_attribute('lower_fov', '-30.0')
        #lidar_bp.set_attribute('horizontal_fov', '30')
        lidar_bp.set_attribute('channels', '64.0')
        lidar_bp.set_attribute("sensor_tick", str(1.0 / SENSOR_FPS))
        lidar_bp.set_attribute('rotation_frequency', str(GLOBAL_FPS))
        lidar_bp.set_attribute('points_per_second', '500000')
        lidar_bp.set_attribute('dropoff_general_rate', '0.0')
        lidar_bp.set_attribute('dropoff_intensity_limit', '1.0')
        lidar_bp.set_attribute('dropoff_zero_intensity', '0.0')
        
        #---------------------------------------------------------------------------
        # Spawn actor

        # Vehicle
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
        npc1 = world.try_spawn_actor(npc_bp, spawn_point_2)
        npc2 = world.try_spawn_actor(npc_bp, spawn_point_3)

        # We will aslo set up the spectator so we can see what we do
        spectator = world.get_spectator()

        # Set viewpoint at main actor
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        # Cam sensor
        spawn_cam_point = carla.Transform(carla.Location(x=1, z=1.5))
        cam_sensor = world.spawn_actor(cam_bp, spawn_cam_point, attach_to=vehicle)

        # Lidar sensor
        spawn_lidar_point = carla.Transform(carla.Location(z=2.5))
        lidar_sensor = world.spawn_actor(lidar_bp, spawn_lidar_point, attach_to=vehicle)


        #----------------------------------------------------------------------------
        # Setting for Traffic manager 
        # Set route for auto pilot
        traffic_manager.set_path(vehicle, route_1)
        
        # Set maximum speed 
        traffic_manager.global_percentage_speed_difference(60)

        # Ignore_Lights
        traffic_manager.ignore_lights_percentage(vehicle, 100)


        #----------------------------------------------------------------------------------------------------------
        # *******CONTROL START HERE******

        object_detection_thread = threading.Thread(target=object_detection)
        object_detection_thread.start()

        lidar_thread = threading.Thread(target=lidar_pointcloud)
        lidar_thread.start()

        # Set Angle and Speed for car
        manual_mode = False
        auto_mode = True
        count = 100
        # In synchronous mode, we need to run the simulation to fly the spectator
        while True:

            #set viewpoint at main actor
            transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
            spectator.set_transform(transform)
            world.tick()
            
            if (manual_mode and count != 0):
                print("tick: ", count, "speed: ", vehicle.get_acceleration())
                if (count >= 41):
                    vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=0))
                    count = count - 1
                elif (count >= 21 and count < 41):
                    vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=-0.35))
                    count = count - 1
                elif (count > 0 and count < 21):
                    vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=0.35))
                    count = count - 1
            else: 
                manual_mode = False
                vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
                vehicle.set_autopilot(auto_mode) # Give TM control over vehicle

            if keyboard.is_pressed('q'):
                lidar_sensor.stop()
                lidar_sensor.destroy()
                cam_sensor.stop()
                cam_sensor.destroy()
                for actor in world.get_actors().filter('*vehicle*'):
                    actor.destroy()
                sys.exit()
            # cv2.imshow('RGB Camera', camera_data['image'])
            # # Break if user presses 'q'
            # if cv2.waitKey(1) == ord('q'):
            #     break

        # #--------------------------------------------------------------------------
        # # Close displayws and stop sensors
        # cv2.destroyAllWindows()
        # lidar_sensor.stop()
        # lidar_sensor.destroy()
        # cam_sensor.stop()
        # cam_sensor.destroy()
        # vis.destroy_window()
        # for actor in world.get_actors().filter('*vehicle*'):
        #     actor.destroy()

        # *******CONTROL END HERE******
        #----------------------------------------------------------------------------