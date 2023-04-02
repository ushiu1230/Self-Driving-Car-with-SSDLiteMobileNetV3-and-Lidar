import carla 
import math
import random
import numpy as np 
import threading
import time
import cv2

#------------------------------------------------------------
IM_HEIGHT = 320
IM_WIDTH = 320 

def camera_callback(image, data_dict):
    data_dict['image'] = np.reshape(np.copy(image.raw_data), (IM_HEIGHT, IM_WIDTH, 4))

def lidar_callback(point_cloud):
        disp_size = [320, 320]
        print(disp_size, lidar_bp.get_attribute('range'))
        lidar_range = 2.0*float(lidar_bp.get_attribute('range'))

        points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)

        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)


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
        settings.fixed_delta_seconds = 0.06
        world.apply_settings(settings)

        # Set up the TM in synchronous mode
        traffic_manager = client.get_trafficmanager()
        traffic_manager.set_synchronous_mode(True)

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
        cam_bp = bp_lib.find("sensor.camera.rgb")
        lidar_bp = bp_lib.find('sensor.lidar.ray_cast')

        print(vehicle_bp)
        print(npc_bp)

        # Set Camera Attribute
        cam_bp.set_attribute("image_size_x",f"{IM_WIDTH}")
        cam_bp.set_attribute("image_size_y",f"{IM_HEIGHT}")
        #cam_bp.set_attribute("fov", "110")
        camera_data = {'image': np.zeros((IM_HEIGHT, IM_WIDTH, 3))}

        # Set Lidar Attribute

        lidar_bp.set_attribute('channels', '64')
        lidar_bp.set_attribute('range', '100')
        lidar_bp.set_attribute('points_per_second', '250000')
        lidar_bp.set_attribute('upper_fov', '5')
        lidar_bp.set_attribute('lower_fov', '-5')
        lidar_bp.set_attribute('rotation_frequency', '20')
        lidar_bp.set_attribute('dropoff_general_rate', lidar_bp.get_attribute('dropoff_general_rate').recommended_values[0])
        lidar_bp.set_attribute('dropoff_intensity_limit', lidar_bp.get_attribute('dropoff_intensity_limit').recommended_values[0])
        lidar_bp.set_attribute('dropoff_zero_intensity', lidar_bp.get_attribute('dropoff_zero_intensity').recommended_values[0])
        
        #---------------------------------------------------------------------------
        # Spawn actor
        # Vehicle
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
        npc1 = world.try_spawn_actor(npc_bp, spawn_point_2)
        npc2 = world.try_spawn_actor(npc_bp, spawn_point_3)

        # Cam sensor
        spawn_cam_point = carla.Transform(carla.Location(x=1, z=1.5))
        cam_sensor = world.spawn_actor(cam_bp, spawn_cam_point, attach_to=vehicle)

        # Lidar sensor
        spawn_lidar_point = carla.Transform(carla.Location(x=0, z=2.4))
        cam_sensor = world.spawn_actor(cam_bp, spawn_lidar_point, attach_to=vehicle)

        #----------------------------------------------------------------------------
        # Setting for Traffic manager 
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
        count = 100

        #----------------------------------------------------------------------------------------------------------
        cam_sensor.listen(lambda image: camera_callback(image, camera_data))

        # *******CONTROL******
        # In synchronous mode, we need to run the simulation to fly the spectator
        while True:
            world.tick()

            #set viewpoint at main actor
            transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
            spectator.set_transform(transform)

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

