import carla 
import math
import random
import time
import numpy as np


def main(): 
    #----------------------------------------------------------------------------------------------------------
    # SET UP CONNECTION 
    # Connect to the client and retrieve the world object
    client = carla.Client('192.168.1.9', 2000)
    world = client.get_world()

    # Set up the simulator in synchronous mode
    settings = world.get_settings()
    settings.synchronous_mode = True # Enables synchronous mode
    settings.fixed_delta_seconds = 0.04
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
    for i, spawn_point in enumerate(spawn_points):
        world.debug.draw_string(spawn_point.location, str(i), life_time=10)

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
    # get blueprint
    bp_lib = world.get_blueprint_library()
    vehicle_bp = bp_lib.filter("model3")[0]
    npc_bp = bp_lib.filter("cooper_s")[0]
    print(vehicle_bp)
    print(npc_bp)
    
    # Spawn actor
    vehicle = world.try_spawn_actor(vehicle_bp, spawn_point_1)
    npc1 = world.try_spawn_actor(npc_bp, spawn_point_2)
    npc2 = world.try_spawn_actor(npc_bp, spawn_point_3)

    # Set view on our Actor
    transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
    spectator.set_transform(transform)

    # Set route for auto pilot
    traffic_manager.set_path(vehicle, route_1)
    

    # Set maximum speed 
    traffic_manager.global_percentage_speed_difference(75)

    # Ignore_Lights
    traffic_manager.ignore_lights_percentage(vehicle, 100)
    

    vehicle.apply_physics_control(carla.VehiclePhysicsControl(max_rpm = 100.0, center_of_mass = carla.Vector3D(0.0, 0.0, 0.0), torque_curve=[[0,40],[100,40]]))


    # Set Angle and Speed for car



    manual_mode = True
    auto_mode = True
    count = 300

    #----------------------------------------------------------------------------------------------------------
    # *******CONTROL******
    # In synchronous mode, we need to run the simulation to fly the spectator
    while True:
        world.tick()

        #set viewpoint at main actor
        transform = carla.Transform(vehicle.get_transform().transform(carla.Location(x=-4,z=2.5)),vehicle.get_transform().rotation)
        spectator.set_transform(transform)
        if (manual_mode and count != 0):
            print("tick: ", count, "speed: ", vehicle.get_acceleration())
            if (count >= 160):
                vehicle.apply_control(carla.VehicleControl(throttle=0.15, steer=0))
                count = count - 1
            elif (count >= 80 and count < 160):
                vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=-0.3))
                count = count - 1
            elif (count > 0 and count < 80):
                vehicle.apply_control(carla.VehicleControl(throttle=0.1, steer=0.3))
                count = count - 1
        else: 
            manual_mode = False
            vehicle.apply_control(carla.VehicleControl(throttle=0, steer=0))
    
        # vehicle.set_autopilot(auto_mode) # Give TM control over vehicle
       
        
if __name__ == "__main__":
    try:
        main()
        
    except KeyboardInterrupt:
        print(' - Exited by user.')
