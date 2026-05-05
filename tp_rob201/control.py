""" A set of robotics control functions """

import random
import numpy as np

random.seed(0)  # for reproducibility of random behavior in obstacle avoidance

def reactive_obst_avoid(lidar, state, rotation_angle, range):
    """
    Reactive controler for ostacle avoidance, using lidar data
    state : 1 for straight line and 0 for turn
    rotation_angle : 1 for left and -1 for right
    range : half of range rotation
    """
    laser_dist = lidar.get_sensor_values() # lidar distances between -180 and 180 degrees 
    
    if np.min(laser_dist[180-range:180+range]) < 20:
        if state == 1:
            state = 0
            rotation_angle = random.choice([-1, 1])  # change rotation direction at each obstacle encounter
            range = random.randint(30, 90)  # change the range of the front sector to consider for obstacle detection
        rotation_speed = 0.4 * rotation_angle
        speed = 0.0
        
    else:
        state = 1
        rotation_speed = 0.0
        speed = 0.5

    return {"forward": speed, "rotation": rotation_speed}, state, rotation_angle, range

def wall_following(lidar, target_dist, Kp, following_state):
    """
    Wall following algoritm inspired by the TP in link of the subject
    """
    
    forward_limit = 30
    
    laser_dist = lidar.get_sensor_values()
    b = laser_dist[90]  # right distance
    a = laser_dist[135]  # front-right distance
    theta = np.radians(45)  # angle between the two lidar measurements (45 degrees)
    localhead = 40 
    
    alpha = np.arctan2(a * np.cos(theta) - b, a * np.sin(theta))  # angle to the wall
    Dt = b * np.cos(alpha)  # distance to the wall
    Dt1 = Dt + localhead * np.sin(alpha)  # distance to the wall at a look-ahead point
    
    error = target_dist - Dt1  
    if abs(error) < 10:
        following_state = "following"
    
    front_dist = np.min(laser_dist[160:200])
    if front_dist < forward_limit:
    # Obstacle in front : turn left to avoid
        command = {"forward": 0.0, "rotation": 0.5}
        return command, following_state
    
    rotation_speed = Kp * error
    # Adaptation of the speed to stay close to the wall
    abs_rot = abs(rotation_speed)
    if abs_rot < 0.1:
        forward_speed = 0.35
    elif abs_rot < 0.3:
        forward_speed = 0.245
    else:
        forward_speed = 0.07
    
    rotation_speed = np.clip(rotation_speed, -1.0, 1.0)
    command = {"forward": forward_speed, "rotation": rotation_speed}
    
    if following_state == "search":
        command = {"forward": 0.4, "rotation": 0.15}  
    return command, following_state

def potential_field_control(lidar, current_pose, goal_pose):
    
    laser_dist = lidar.get_sensor_values()
    ray_angles = lidar.get_ray_angles()
    
    seuil   = 80
    epsilon = 10
    d_safe  = 50
    K_obs   = 200
    K_goal  = 1.0
    K_rot   = 0.6
    agle_max = np.pi / 4

    

    # Repulsive gradient 
    obstacle_grad = np.array([0.0, 0.0])
    

    for i in range(len(laser_dist)):
        d = laser_dist[i]
        if d < d_safe:
            angle_abs = current_pose[2] + ray_angles[i]
            obs_pos = current_pose[:2] + np.array([
                d * np.cos(angle_abs),
                d * np.sin(angle_abs)
            ])
            direction = current_pose[:2] - obs_pos      
            obstacle_grad += (K_obs / d**3) * (1/d - 1/d_safe) * direction

    # Attractive gradient
    dist_to_goal = np.linalg.norm(goal_pose[:2] - current_pose[:2])

    if dist_to_goal < epsilon:
        return {"forward": 0.0, "rotation": 0.0}

    if dist_to_goal > seuil:
        K_mv = 0.3
        attract_grad = K_goal * (goal_pose[:2] - current_pose[:2]) / dist_to_goal
    else:
        K_mv = 0.008
        attract_grad = K_goal * (goal_pose[:2] - current_pose[:2])

    # Total gradient
    total_grad = attract_grad + obstacle_grad
    total_norm = np.linalg.norm(total_grad)

    # Moving command
    angle = np.arctan2(total_grad[1], total_grad[0]) - current_pose[2]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi   # normalisation [-pi, pi]

    rotation_speed = np.clip(K_rot * angle / np.pi, -1.0, 1.0)

    if abs(angle) > agle_max:
        forward_speed = K_mv * total_norm * (agle_max / abs(angle))
    else:
        forward_speed = K_mv * total_norm
    forward_speed = np.clip(forward_speed, 0.05, 1.0)

    forward_speed = np.clip(forward_speed, 0.0, 1.0)

    return {"forward": forward_speed, "rotation": rotation_speed}

def random_free_goal(self):
    """Chose a random goal position in a free area."""
    x_min = -(1013/2 + 389.0)
    x_max =  1013/2 - 389.0
    y_min = -(700/2 + 170.0)
    y_max =  700/2 - 170.0

    for _ in range(200):
        x = np.random.uniform(x_min, x_max)
        y = np.random.uniform(y_min, y_max)
        ix, iy = self.occupancy_grid.conv_world_to_map(x, y)
        grid = self.occupancy_grid.occupancy_map
        if 0 <= ix < grid.shape[0] and 0 <= iy < grid.shape[1]:
            if grid[ix, iy] < -30.0:
                return np.array([x, y, 0.0])


    return np.array([0.0, 0.0, 0.0])