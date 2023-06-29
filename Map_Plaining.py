import carla
import random

# Connect to the client and retrieve the world object
client = carla.Client('localhost', 2000)
world = client.get_world()

# Set up the simulator in synchronous mode
settings = world.get_settings()
settings.synchronous_mode = True # Enables synchronous mode
settings.fixed_delta_seconds = 0.01
world.apply_settings(settings)

# Set up the TM in synchronous mode
traffic_manager = client.get_trafficmanager()
traffic_manager.set_synchronous_mode(True)

# Set a seed so behaviour can be repeated if necessary
traffic_manager.set_random_device_seed(0)
random.seed(0)

# We will aslo set up the spectator so we can see what we do
spectator = world.get_spectator()


spawn_points = world.get_map().get_spawn_points()


# Draw the spawn point locations as numbers in the map
for i, spawn_point in enumerate(spawn_points):
    world.debug.draw_string(spawn_point.location, str(i), life_time=1000)

# In synchronous mode, we need to run the simulation to fly the spectator
while True:
    world.tick()