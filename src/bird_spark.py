import os
os.system('clear')
import numpy as np
import time
from pyspark.sql import SparkSession
import argparse

from get_gif import *

def compute_speed(velocity):
    return np.linalg.norm(velocity)

def limit_speed(velocity, min_speed, max_speed):
    speed = compute_speed(velocity)

    if speed < 1e-10:  
        return np.zeros_like(velocity)

    if speed < min_speed:
        velocity = velocity / speed * min_speed
    elif speed > max_speed:
        velocity = velocity / speed * max_speed

    return velocity


def update_lead_bird_position(t):

    angle = lead_bird_speed * t / lead_bird_radius  
    x = lead_bird_radius * np.cos(angle)
    y = lead_bird_radius * np.sin(angle) * np.cos(angle)
    z = lead_bird_radius * (1 + 0.5 * np.sin(angle / 5))

    return np.array([x, y, z])




def compute_forces(bird_position, positions):
    distances = np.linalg.norm(positions - bird_position, axis=1)

    d_lead = distances[0]
    lead_force = (positions[0] - bird_position) * ((1 / (d_lead))) if d_lead > 10 else np.zeros(3)

    nearest_idx = np.argmin(distances)
    d_near = distances[nearest_idx]
    cohesion_force = np.nan_to_num((positions[nearest_idx] - bird_position) * ((d_near / 1) ** 2)) if d_near > max_distance else np.zeros(3)

    close_neighbors = positions[distances < min_distance]
    close_distances = distances[distances < min_distance]
    separation_force = np.sum([(bird_position - neighbor) / (dist ** 2)
                         for neighbor, dist in zip(close_neighbors, close_distances) if dist > 0],
                        axis=0) if len(close_neighbors) > 0 else np.zeros(3)

    total_weight = np.sum([1 / ((dist / 1) ** 2) for dist in close_distances if dist > 0])
    if total_weight > 0:
        separation_force = separation_force / total_weight

    return cohesion_force + separation_force + lead_force




def update_positions(indices):
    updated_data = []
    for i in indices:
        bird_position = positions_broadcast.value[i]
        velocity = velocities_broadcast.value[i]
        velocity += compute_forces(bird_position, positions_broadcast.value)
        velocity = limit_speed(velocity, min_speed, max_speed)
        bird_position += velocity * time_step
        updated_data.append((i, bird_position, velocity))
    return updated_data






import os
if __name__ == "__main__":

    outputDir = "./plot"
    if not os.path.exists(outputDir):
        os.makedirs(outputDir)


    parser = argparse.ArgumentParser(description="Edit Distance with PySpark")
    parser.add_argument('--num_birds', type=int, default=10000, help="Number of birds")
    args = parser.parse_args()

    numBirds = args.num_birds
    numFrames = 500

    time_step = 1 / 4

    stdDevPosition = 10.0
    lead_bird_speed = 20.0
    lead_bird_radius = 300.0
    min_speed = 10.0
    max_speed = 30.0
    max_distance = 20.0
    min_distance = 10.0

    positions = np.random.normal(loc=np.array(
        [0, 0, 1.5 * lead_bird_radius]),
                                 scale=stdDevPosition,
                                 size=(numBirds, 3))
    velocities = np.zeros((numBirds, 3))

    spark = SparkSession.builder \
        .appName("BirdFlockSimulation") \
        .master("local[*]") \
        .config("spark.executor.memory", "4g") \
        .getOrCreate()
    sc = spark.sparkContext

    simulation = []
    timeCost = []

    for frame in range(numFrames):
        start = time.time()

        positions[0] = update_lead_bird_position(frame * time_step)

        positions_broadcast = sc.broadcast(positions)
        velocities_broadcast = sc.broadcast(velocities)

        bird_indices = sc.parallelize(range(1, numBirds))

        updated_data = bird_indices.mapPartitions(
            lambda indices: update_positions(list(indices))
        ).collect()

        for idx, new_position, new_velocity in updated_data:
            positions[idx] = new_position
            velocities[idx] = new_velocity

        end = time.time()
        frameCost = end - start
        timeCost.append(frameCost)

        simulation.append(positions.copy())
        print(f'frame simulation time: {frameCost:.4f}s')

    meanTime = np.mean(timeCost)
    print(f'Average time cost per frame: {meanTime:.4f}')

    spark.stop()

    visualize_simulation(simulation, lead_bird_radius)
    create_compressed_gif("./plot", gif_name="bird_simulation.gif", duration=100, loop=1, resize_factor=0.5)