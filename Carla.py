#!/usr/bin/env python

# Copyright (c) 2020 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

"""
Script that render multiple sensors in the same pygame window

By default, it renders four cameras, one LiDAR and one Semantic LiDAR.
It can easily be configure for any different number of sensors. 
To do that, check lines 290-308.
"""

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import math
import carla
import argparse
import random
import time
import numpy as np
import cv2
import torch
import queue
import threading

from utils.coco_names import coco_names
from models.Model import *
from utils.detect_utils import *

#------------------------------------------------------------
# Define queue to store frames
image_queue = queue.Queue(maxsize=1)
lidar_queue = queue.Queue(maxsize=1)
Lidar_result_queue = queue.Queue(maxsize=1)
Camera_result_queue = queue.Queue(maxsize=1)

# Global variables
Start_Sig = False
is_obstacle_found = False

#------------------------------------------------------------
# Define model
device = torch.device('cuda')
model = Model4(device)

try:
    import pygame
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_q
    from pygame.locals import K_e
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

def detect_objects(image):
    global is_obstacle_found
    start_time = time.time()
    boxes, classes, labels = predict(image, model, device, 0.9)

    for box, pred_class, label in zip(boxes, classes, labels):
        if box is not None:
            is_obstacle_found = True
        else:
            is_obstacle_found = False
    # get predictions for the current frame  
    # draw boxes
    frame = draw_boxes(boxes, classes, labels, image)
    fps = 1/(time.time() - start_time)
    # write the FPS on the current frame
    cv2.putText(frame, f"{fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    #convert from BGR to RGB color format
    # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return frame

class ImageProcessorThread(threading.Thread):
    def __init__(self, image_queue, display_man, display_pos):
        super(ImageProcessorThread, self).__init__()
        self.image_queue = image_queue
        self.display_man = display_man
        self.display_pos = display_pos
        self.surface = None

    def run(self):
        global is_obstacle_found
        while True:
            image = self.image_queue.get()
            image = detect_objects(image)

            self.surface = pygame.surfarray.make_surface(image.swapaxes(0, 1))
            if self.surface is not None:
                offset = self.display_man.get_display_offset(self.display_pos)
                self.display_man.display.blit(self.surface, offset)
            self.image_queue.task_done()

class LidarProcessorThread(threading.Thread):
    def __init__(self, lidar_queue, Lidar_result_queue):
        super(LidarProcessorThread, self).__init__()
        self.lidar_queue = lidar_queue
        self.Lidar_result_queue = Lidar_result_queue

    def run(self):
        while True:
            points = self.lidar_queue.get()
            points[:, :1] = -points[:, :1]
            Sensor_loc = carla.Location(0,0,2.5)
            delta_x = np.power(points[:, 0] - Sensor_loc.x, 2)
            delta_y = np.power(points[:, 1] - Sensor_loc.y, 2)
            # delta_z = np.power(points[:, 2] - Sensor_loc.z, 2)
          
            distances = np.sqrt(delta_x + delta_y)

            # Find the index of the minimum distance
            min_index = np.argmin(distances)

            # Get the minimum distance value
            min_distance = distances[min_index]
            min_point = points[min_index, :2]
            result = [min_distance, min_point]
            Lidar_result_queue.put(result)
            self.lidar_queue.task_done()

class StateMachine(threading.Thread):
    def __init__(self, Lidar_result_queue, vehicle, tm):
        super(StateMachine, self).__init__()
        self.current_state = "Start"
        self.previous_state = "Start"
        self.Lidar_result_queue = Lidar_result_queue
        self.vehicle = vehicle
        self.tm = tm
        self.count = 60

    def brake_on_distance(self, distance, x_min, x_max, y_min, y_max):
        return ((distance - x_min) / (x_max - x_min)) * (y_max - y_min) + y_min

    def transition(self):
        global is_obstacle_found
            
        Lidar = self.Lidar_result_queue.get()
        distance = Lidar[0]
        x =  Lidar[1][0]
        y = Lidar[1][1]

        if self.current_state == "Normal":
            # precondition
            if not self.previous_state == "Normal":
                print("previous_state Change")
                self.vehicle.set_autopilot(True)
                self.tm.global_percentage_speed_difference(0)

            self.previous_state = self.current_state

            if is_obstacle_found:
                if (8 < distance <= 12) and -12 < x < 0 and (-0.2 <= y <= 0.2):
                    self.tm.global_percentage_speed_difference(50)
                    self.current_state = "Slow Down"

                elif (1 < distance <= 6) and x < 0 and (-0.2 <= y <= 0.2):
                    self.tm.global_percentage_speed_difference(50)
                    self.current_state = "Stop"
            
            elif (8 < distance <= 12) and -12 < x < 0 and (-1  <= y <= 1):
                self.tm.global_percentage_speed_difference(50)
                self.current_state = "Slow Down"

            elif (1 < distance <= 6) and (-1 <= y <= 1):
                self.tm.global_percentage_speed_difference(50)
                self.current_state = "Stop"

        elif self.current_state == "Slow Down":
            # precondition 
            if not self.previous_state == "Slow Down":
                self.vehicle.set_autopilot(False)

            self.previous_state = self.current_state

            brake_offset = self.brake_on_distance(distance, 12, 8, 0.6, 0.8)
            self.vehicle.apply_control(carla.VehicleControl(throttle=0.2, steer=0.0, brake=brake_offset))

            if is_obstacle_found:
                if (1 < distance <= 6) and x < 0 and (-0.2 <= y <= 0.2):
                    self.current_state = "Stop" 

                elif (6 < distance <= 8) and  -9 < x < 0 and (-0.2 <= y <= 0.2):
                    self.current_state = "Pre: Change lane"

        elif self.current_state == "Pre: Change lane":
            # precondition
            if not self.previous_state == "Pre: Change lane":
                self.vehicle.set_autopilot(False)

            self.previous_state = self.current_state

            self.vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=-0.6, brake=0))

            #print("count: ",self.count)

            if self.count != 0: 
                self.count = self.count - 1

            else:
                # if not is_obstacle_found and (distance > 2):
                    self.count = 70
                    self.current_state = "Pre: Pre Steering"


        elif self.current_state == "Pre: Pre Steering":
            if not self.previous_state == "Pre: Pre Steering":
                self.vehicle.set_autopilot(False)

            self.previous_state = self.current_state

            self.vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=0.6, brake=0))

            #print("count: ",self.count)

            if self.count != 0: 
                self.count = self.count - 1

            else:
            # if not is_obstacle_found and (distance > 2):
                self.count = 100
                self.tm.global_percentage_speed_difference(20)
                self.current_state = "Holding"

        elif self.current_state == "Holding":
            if not self.previous_state == "Holding":
                self.vehicle.set_autopilot(True)

            self.previous_state = self.current_state

            print("count: ",self.count)

            if self.count != 0:
                if not (distance <= 3 and (x > 0 and y < 0)):
                    self.count = self.count - 1

            else:
                if distance > 3 and not(x > 0 and y < 0):
                    #and (x > 0) and (-0.2 <= y <= 0.2) and not is_obstacle_found
                    self.count = 60
                    self.current_state = "Pos: Change back lane"

        elif self.current_state == "Pos: Change back lane":
            if not self.previous_state == "Pos: Change back lane":
                print("previous_state Change")
                self.tm.global_percentage_speed_difference(0)
                self.vehicle.set_autopilot(False)


            self.previous_state = self.current_state

            self.vehicle.apply_control(carla.VehicleControl(throttle=0.5, steer=0.6, brake=0))

            if self.count != 0: 
                self.count = self.count - 1

            else:
            # if not is_obstacle_found and (distance > 2):
                self.count = 70
                self.current_state = "Pos: Return Steering"


        elif self.current_state == "Pos: Return Steering":
            if not self.previous_state == "Pos: Return Steering":
                print("previous_state Change")
                self.vehicle.set_autopilot(False)

            self.previous_state = self.current_state

            self.vehicle.apply_control(carla.VehicleControl(throttle=0.35, steer=-0.6, brake=0))

            #print("count: ",self.count)

            if self.count != 0: 
                self.count = self.count - 1

            else:
            # if not is_obstacle_found and (distance > 2):
                self.count = 60
                self.current_state = "Normal"



        elif self.current_state == "Stop":
            if not self.previous_state == "Stop":
                self.vehicle.set_autopilot(False)

            self.previous_state = self.current_state

            brake_offset = self.brake_on_distance(distance, 6, 3, 0.8, 1)
            self.vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0.0, brake=brake_offset))

            if not is_obstacle_found:
                if  6 < distance <= 9:
                    self.current_state = "Slowdown"
        
        # Get the vehicle's velocity
        vehicle_velocity = self.vehicle.get_velocity()

        # Compute the vehicle speed
        speed = 3.6 * (vehicle_velocity.x**2 + vehicle_velocity.y**2 + vehicle_velocity.z**2)**0.5

        print(f"State: {self.current_state}, Distance: {distance}, Speed: {speed}")

        
    def run(self): 
        global Start_Sig
        while True:
            if self.current_state == "Start":
                if Start_Sig:
                    self.current_state = "Normal"
            self.transition()


class CustomTimer:
    def __init__(self):
        try:
            self.timer = time.perf_counter
        except AttributeError:
            self.timer = time.time

    def time(self):
        return self.timer()

class DisplayManager:
    def __init__(self, grid_size, window_size):
        pygame.init()
        pygame.font.init()
        self.display = pygame.display.set_mode(window_size, pygame.HWSURFACE | pygame.DOUBLEBUF)

        self.grid_size = grid_size
        self.window_size = window_size
        self.sensor_list = []

    def get_window_size(self):
        return [int(self.window_size[0]), int(self.window_size[1])]

    def get_display_size(self):
        return [int(self.window_size[0]/self.grid_size[1]), int(self.window_size[1]/self.grid_size[0])]

    def get_display_offset(self, gridPos):
        dis_size = self.get_display_size()
        return [int(gridPos[1] * dis_size[0]), int(gridPos[0] * dis_size[1])]

    def add_sensor(self, sensor):
        self.sensor_list.append(sensor)

    def get_sensor_list(self):
        return self.sensor_list

    def render(self):
        if not self.render_enabled():
            return

        for s in self.sensor_list:
            s.render()

        pygame.display.flip()

    def destroy(self):
        for s in self.sensor_list:
            s.destroy()

    def render_enabled(self):
        return self.display != None

class SensorManager:
    def __init__(self, world, queue, display_man, sensor_type, transform, attached, sensor_options, display_pos):
        self.surface = None
        self.world = world
        self.queue = queue
        self.display_man = display_man
        self.display_pos = display_pos
        self.sensor = self.init_sensor(sensor_type, transform, attached, sensor_options)
        self.sensor_options = sensor_options
        self.timer = CustomTimer()

        self.time_processing = 0.0
        self.tics_processing = 0

        self.display_man.add_sensor(self)

    def init_sensor(self, sensor_type, transform, attached, sensor_options):
        if sensor_type == 'RGBCamera':
            camera_bp = self.world.get_blueprint_library().find('sensor.camera.rgb')
            disp_size = self.display_man.get_display_size()
            camera_bp.set_attribute('image_size_x', f"{256}")
            camera_bp.set_attribute('image_size_y', f"{256}")

            for key in sensor_options:
                camera_bp.set_attribute(key, sensor_options[key])

            camera = self.world.spawn_actor(camera_bp, transform, attach_to=attached, attachment_type = carla.AttachmentType.Rigid)
            camera.listen(self.save_rgb_image)

            return camera

        elif sensor_type == 'LiDAR':
            lidar_bp = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
            lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
            lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
            lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])

            for key in sensor_options:
                lidar_bp.set_attribute(key, sensor_options[key])

            lidar = self.world.spawn_actor(lidar_bp, transform, attach_to=attached)

            lidar.listen(self.save_lidar_image)

            return lidar
        else:
            return None

    def get_sensor(self):
        return self.sensor

    def save_rgb_image(self, image):
        t_start = self.timer.time()
        image.convert(carla.ColorConverter.Raw)
        array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
        array = np.reshape(np.copy(image.raw_data), (image.height, image.width, 4))
        array = array[:, :, :3]
        self.queue.put(array)
        array = array[:, :, ::-1]
        
        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1

    def save_lidar_image(self, image):
        t_start = self.timer.time()

        disp_size = self.display_man.get_display_size()
        lidar_range = 2.0*float(self.sensor_options['range'])

        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))

        lidar_data_temp = np.array(points[:, :3])

        if np.any((lidar_data_temp[:, 0] > 0) & (lidar_data_temp[:, 0] <= 15) & (lidar_data_temp[:, 1] < 0.5) & (lidar_data_temp[:, 1] > -0.5) & (lidar_data_temp[:, 2] < 0)):
            print("At least one point satisfies the conditions.")
        else:
            print("No point satisfies the conditions.")


        lidar_data = np.array(points[:, :2])

        self.queue.put(np.copy(lidar_data))
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        if self.display_man.render_enabled():
            self.surface = pygame.surfarray.make_surface(lidar_img)

        t_end = self.timer.time()
        self.time_processing += (t_end-t_start)
        self.tics_processing += 1
        
    def render(self):
        if self.surface is not None:
            offset = self.display_man.get_display_offset(self.display_pos)
            self.display_man.display.blit(self.surface, offset)

    def destroy(self):
        self.sensor.destroy()

def run_simulation(args, client):
    """This function performed one test run using the args parameters
    and connecting to the carla client passed.
    """

    display_manager = None
    vehicle = None
    vehicle_list = []
    timer = CustomTimer()
    try:

        # Getting the world and
        world = client.get_world()
        original_settings = world.get_settings()
        #----------------------------------------------------------------------------------------------------------
        # GET SOME SPAWN POINT ON MAP AND DEFINE THE AUTOPILOT ROUTE

        spawn_points = world.get_map().get_spawn_points()
        # # Draw the spawn point locations as numbers in the map
        # for i, spawn_point in enumerate(spawn_points):
        #     world.debug.draw_string(spawn_point.location, str(i), life_time=10 )

        # Route 1
        spawn_point_1 =  spawn_points[60]
        spawn_point_2 =  spawn_points[10]
        spawn_point_3 =  spawn_points[47]
        spawn_point_4 =  spawn_points[51]


        # Create route 1 from the chosen spawn points
        route_1_indices = [149, 21, 105, 52, 104, 140, 10]
        route_1 = []
        for ind in route_1_indices:
            route_1.append(spawn_points[ind].location)

        # Create route 2 from the chosen spawn points
        route_2_indices = [103, 111, 115, 140]
        route_2 = []
        for ind in route_2_indices:
            route_2.append(spawn_points[ind].location)

        # # Now let's print them in the map so we can see our routes
        # world.debug.draw_string(spawn_point_1.location, 'Spawn point 1', life_time=30, color=carla.Color(255,0,0))
            
        # for ind in route_1_indices:
        #     spawn_points[ind].location
        #     world.debug.draw_string(spawn_points[ind].location, str(ind), life_time=60, color=carla.Color(255,0,0))

        if args.sync:
            print("sync mode on")
            traffic_manager = client.get_trafficmanager(8000)
            settings = world.get_settings()
            traffic_manager.set_synchronous_mode(True)
            settings.synchronous_mode = True
            settings.fixed_delta_seconds = 0.01
            world.apply_settings(settings)

        #---------------------------------------------------------------------------
        # Spawn actor
        
        # Instanciating the vehicle to which we attached the sensors
        bp = world.get_blueprint_library()

        vehicle_bp = bp.filter("model3")[0]
        npc_bp = bp.filter("cooper_s")[0]

        # Vehicle
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
        Static_car1 = world.try_spawn_actor(npc_bp, spawn_point_2)
        Static_car2 = world.try_spawn_actor(npc_bp, spawn_point_3)
        Moving_car1 = world.try_spawn_actor(npc_bp, spawn_point_4)
        
        vehicle_list.append(vehicle)
        vehicle_list.append(Static_car1)
        vehicle_list.append(Static_car2)
        vehicle_list.append(Moving_car1)

        # We will aslo set up the spectator so we can see what we do
        spectator = world.get_spectator()
        
        # Set viewpoint at main actor
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
        spectator.set_transform(transform)

        #----------------------------------------------------------------------------
        # Setting for Traffic manager 
        # Set route for auto pilot
        traffic_manager.set_path(vehicle, route_1)
        traffic_manager.set_path(Moving_car1, route_2)

        time.sleep(0.5)

        # Ignore_Lights
        # for car in vehicle_list:
        #     print(car)
        traffic_manager.ignore_lights_percentage(vehicle, 100)
        traffic_manager.ignore_vehicles_percentage(vehicle, 100)

        # Display Manager organize all the sensors an its display in a window
        # If can easily configure the grid and the total window size
        display_manager = DisplayManager(grid_size=[1, 3], window_size=[args.width, args.height])

        # Then, SensorManager can be used to spawn RGBCamera, LiDARs and SemanticLiDARs as needed
        # and assign each of them to a grid position, 
        SensorManager(world, image_queue, display_manager, 'RGBCamera', carla.Transform(carla.Location(x=1, z=1.5)), 
                      vehicle, {}, display_pos=[0, 0])
        SensorManager(world, lidar_queue, display_manager, 'LiDAR', carla.Transform(carla.Location(x=0, z=1.6)), 
                      vehicle, {'channels' : '64', 'range' : '30', 'upper_fov': '6', 'lower_fov': '-4',  'points_per_second': '1000000', 'rotation_frequency': '100'}, display_pos=[0, 2])

        #Simulation loop
        call_exit = False
        time_init_sim = timer.time()

        # Create a new thread to process the image
        image_processor_thread = ImageProcessorThread(image_queue, display_manager, display_pos=[0, 1])
        lidar_processor_thread = LidarProcessorThread(lidar_queue, Lidar_result_queue)
        Car_Control = StateMachine(Lidar_result_queue, vehicle, traffic_manager)

        # Start the threads
        image_processor_thread.start()
        lidar_processor_thread.start()
        Car_Control.start()

        # Wait for the queue to be empty before continuing with the main thread
        image_queue.join()
        lidar_queue.join()
        Lidar_result_queue.join()
        
        print("Threads are running:", threading.active_count())

        
        while True:
            global Start_Sig


            # Carla Tickpyth
            if args.sync:
                world.tick()
            else:
                world.wait_for_tick()

            # Render received data
            display_manager.render()

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    call_exit = True
                elif event.type == pygame.KEYDOWN:
                    if event.key == K_ESCAPE or event.key == K_q:
                        call_exit = True
                        break
                    if event.key == K_e:
                        Start_Sig = True
            
            if call_exit:
                break
        
    finally:
        if display_manager:
            display_manager.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in vehicle_list])

        world.apply_settings(original_settings)


def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Sensor tutorial')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '--sync',
        action='store_true',
        help='Synchronous mode execution')
    argparser.add_argument(
        '--async',
        dest='sync',
        action='store_false',
        help='Asynchronous mode execution')
    argparser.set_defaults(sync=True)
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='800x600',
        help='window resolution (default: 1280x720)')

    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]


    arr = np.random.rand(255, 255, 3).astype(np.float32)
    boxes, classes, labels = predict(arr, model, device, 0.9)

    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(5.0)

        run_simulation(args, client)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        sys.exit(0)


if __name__ == '__main__':

    main()
