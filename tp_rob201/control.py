""" A set of robotics control functions """

import random
import numpy as np

random.seed(0)  # for reproducibility of random behavior in obstacle avoidance

def reactive_obst_avoid(lidar):
    """
    Simple obstacle avoidance
    lidar : placebot object with lidar data
    """
    # TODO for TP1

    global rot_speed # variable globale pour eviter les comportements ératiques proche des murs

    laser_dist = lidar.get_sensor_values()
    speed = 1.0
    rotation_speed = 0.0
    if np.min(laser_dist[135:225]) < 35:
        rotation_speed = rot_speed  
        speed = 0.0  
    else:
        rot_speed = 0.8* random.choice([-1, 1])  
        rotation_speed = 0.0
        speed = 1.0
    
    command = {"forward": speed,
               "rotation": rotation_speed}
    
    return command

def wall_following(lidar, target_wall_dist, Kp, state, side):

    distances  = lidar.get_sensor_values()
    front_dist = np.min(distances[160:200])
    left_dist  = distances[270]
    right_dist = distances[90]
    
    front_left = distances[225]
    front_right = distances[135]

    if state == "search":
        speed = 0.4
        rotation = 0.0
        if front_dist < target_wall_dist * 1.5:
            speed = 0.0
            if side == "left":
                rotation = -0.5
                if left_dist < target_wall_dist:
                    state = "follow"
            else:
                rotation = 0.5
                if right_dist < target_wall_dist:
                    state = "follow"

    if state == "follow":
        if side == "left":
            if left_dist > target_wall_dist * 2.5 and front_dist > target_wall_dist * 2.5:
                state = "turn_around_corner"          
            else:                                     
                error    = left_dist - target_wall_dist
                diag_error = front_left - target_wall_dist
                combined_error = 0.6* error + 0.5 * diag_error  
                rotation = np.clip(Kp * combined_error, -1.0, 1.0)
                speed    = 0.4 if abs(rotation) < 0.3 else 0.2
                if front_dist < target_wall_dist * 1.2:
                    speed    = 0.2
                    rotation = -0.5
        if side == "right":
            if right_dist > target_wall_dist * 2.5 and front_dist > target_wall_dist * 2.5:
                state = "turn_around_corner"          
            else:                                     
                error    = right_dist - target_wall_dist
                diag_error = front_right - target_wall_dist
                combined_error = 0.6 * error + 0.5 * diag_error 
                rotation = np.clip(-Kp * combined_error, -1.0, 1.0)
                speed    = 0.4 if abs(rotation) < 0.3 else 0.2
                if front_dist < target_wall_dist * 1.2:
                    speed    = 0.2
                    rotation = 0.5

    if state == "turn_around_corner":
        if side == "left":
            speed    = 0.4
            rotation = 0.2
            if left_dist < target_wall_dist * 1.8:
                rotation = 0.0
            if left_dist < target_wall_dist * 1.8:
                state = "follow"
        if side == "right":
            speed    = 0.4
            rotation = -0.2
            if right_dist < target_wall_dist * 2:
                rotation = 0.0
            if right_dist < target_wall_dist * 1.8:
                state = "follow"

    command = {"forward": speed, "rotation": rotation}
    return command, state 

def potential_field_control(lidar, current_pose, goal_pose):
    """
    Control using potential field for goal reaching and obstacle avoidance
    lidar : placebot object with lidar data
    current_pose : [x, y, theta] nparray, current pose in odom or world frame
    goal_pose : [x, y, theta] nparray, target pose in odom or world frame
    Notes: As lidar and odom are local only data, goal and gradient will be defined either in
    robot (x,y) frame (centered on robot, x forward, y on left) or in odom (centered / aligned
    on initial pose, x forward, y on left)
    """
    
    seuil = 60 # distance threshold to pass from linear to quadratic potential
    epsilon = 2  # small value to set a threshold for the potential field to avoid oscillations near the goal
    d_safe = 70  # distance threshold to consider an obstacle as a threat for the robot
    
    
    # Definition of the repulsive potential field from the nearest obstacle
    K_obs = 0.5  # gain for repulsive potential
    laser_dist = lidar.get_sensor_values()
    obstacle_dist = np.min(laser_dist)
    print("Obstacle distance:", obstacle_dist)
    
    obstacle_pose = current_pose[:2] + np.array([
        obstacle_dist * np.cos(current_pose[2]),
        obstacle_dist * np.sin(current_pose[2])
    ])

    direction = current_pose[:2] - obstacle_pose         
    direction = direction / np.linalg.norm(direction) 
    
    if obstacle_dist < d_safe:
        obstacle_grad = (K_obs / (obstacle_dist)**3) * (1 / obstacle_dist - 1 / d_safe) * direction # repulsive potential gradient
    else :
        obstacle_grad = np.array([0.0, 0.0])  # no repulsive force if no obstacle is too close
    
    # Definition of global potential field
    
    K_goal = 1.0  # gain for attractive potential
    linear_grad = K_goal * (goal_pose[:2] - current_pose[:2]) / np.linalg.norm(goal_pose[:2] - current_pose[:2])  # attractive potential gradient
    linear_norm = np.linalg.norm(linear_grad + obstacle_grad) # lineargradient norm
    
    quadratic_grad = K_goal * (goal_pose[:2] - current_pose[:2])# quadratic potential gradient
    quadratic_norm = np.linalg.norm(quadratic_grad + obstacle_grad) # quadratic gradient norm
    

    
    # Displacement command based on potential field gradient
    print(linear_norm, quadratic_norm)
    if quadratic_norm < epsilon:
        return {"forward": 0.0, "rotation": 0.0}  # stop if close enough to the goal
    K_rot = 0.5  # gain for rotation
    K_mv = 0.4  # gain for forward movement
    agle_max = np.pi / 6  # maximum rotation angle (90 degrees)
    
    
    if np.linalg.norm(goal_pose[:2] - current_pose[:2]) > seuil:
        K_mv = 0.4
        
        angle = (np.arctan2(linear_grad[1], linear_grad[0])-current_pose[2])  # angle to goal minus current orientation
        angle = ((angle + np.pi) % (2 * np.pi) - np.pi)  # normalize angle to [-pi, pi]
        rotation_speed = K_rot * angle/np.pi
        
        if np.abs(angle) > agle_max:
            forward_speed = K_mv * linear_norm * (agle_max / angle)  # limit rotation speed
        else:
            forward_speed = K_mv * linear_norm  # move forward proportional to the gradient norm
        forward_speed = min(forward_speed, 1.0)  # limit forward speed to 1.0

            
        
    else :
        K_mv = 0.02
        angle = (np.arctan2(quadratic_grad[1], quadratic_grad[0])-current_pose[2])  # angle to goal minus current orientation
        angle = ((angle + np.pi) % (2 * np.pi) - np.pi)  # normalize angle to [-pi, pi]
        rotation_speed = K_rot * angle/np.pi
    
        if np.abs(angle) > agle_max:
            forward_speed = K_mv * quadratic_norm * (agle_max / angle)  # limit rotation speed
        else:
            forward_speed = K_mv * quadratic_norm  # move forward proportional to the gradient norm
        forward_speed = min(forward_speed, 1.0)  # limit forward speed to 1.0

            
    # Position de l'objectif et du robot dans le repère odométrique / valeurs du lidar dans le repère robot, on peut faire la conversion pour avoir les deux dans le même repère et calculer le gradient de potentiel en fonction des données lidar et de la position de l'objectif
    
    # Choix du but, prendre une direction aléatoire en fonction des données lidar et une distance inférieure à la plus petite valeur du lidar
    command = {"forward": forward_speed,
               "rotation": rotation_speed}

    return command
