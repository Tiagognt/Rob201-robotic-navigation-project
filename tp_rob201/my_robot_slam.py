"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np
import random as rd

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, wall_following, random_free_goal
from occupancy_grid import OccupancyGrid
from planner import Planner


# Definition of our robot controller
class MyRobotSlam(RobotAbstract):
    """A robot controller including SLAM, path planning and path following"""


    def __init__(self,
                 lidar_params: LidarParams = LidarParams(),
                 odometer_params: OdometerParams = OdometerParams()):
        # Passing parameter to parent class
        super().__init__(lidar_params=lidar_params,
                         odometer_params=odometer_params)

        # step counter to deal with init and display
        self.counter = 0
        
        # attribute for TP1 controler
        self.state = 1 # 1 for straight line and 0 for turn
        self.range = 90 # half of range rotation 
        self.rotation_angle = 1 # 1 for left and -1 for right
        
        self.following_state = "search" # state for wall following control
        
        #attribute for TP2 controler
        self.current_goal = [0.0, -95.0]  # initial goal is bellow the starting position
        self.previous_pose = [0.0, 0.0, 0.0]  # to detect if the robot is stuck
        self.stuck_counter = 0

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1113, 750)
        robot_position = (439.0, 195.0)

        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)
        
        # attribute to the planning and path following
        self.exploration_counter = 0  # number of iterations to perform random exploration before planning
        self.plannig = False  
        self.traj = None  # to store the current trajectory to the goal for display
        self.path_counter = 0  # to follow the trajectory point by point

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        

    def control(self):
        """
        Main control function executed at each time step
        """
        
        # Section of the TP3 
        odom_pose = self.odometer_values()
        lidar = self.lidar()
        
        # the TODO section for TP4
        
        init_iterarion = 50  # nombre d'itérations à attendre avant de commencer la localisation
        
        if self.counter < init_iterarion:
            self.tiny_slam.update_map(lidar, odom_pose)
            corrected_pose = odom_pose  # pendant les premières itérations, on utilise la pose
            
        else:
            
            score = self.tiny_slam.localise(lidar, odom_pose)
            # Mise à jour de la carte seulement si le score est bon
            SCORE_THRESHOLD = 50  # à ajuster selon tes résultat
            corrected_pose = self.tiny_slam.get_corrected_pose(odom_pose)
            if score > SCORE_THRESHOLD:
                self.tiny_slam.update_map(lidar, corrected_pose)
        
        
        if self.counter < 2000:  
            command = self.control_tp1() 
        else:
            if not self.plannig:
                self.traj = self.planner.plan(corrected_pose, [0.0,0.0,0.0])  # plan a trajectory to the origin (or any other goal)
                self.plannig = True
            self.current_goal = self.traj[:, self.path_counter]  # update the current goal to the end of the trajectory for display
            command = potential_field_control(lidar, corrected_pose, self.current_goal) 
            if np.linalg.norm(corrected_pose[:2] - self.current_goal) < 20 and self.path_counter < self.traj.shape[1] - 1:
                self.path_counter += 1  # move to the next point in the trajectory at the next iteration
            
        if self.counter % 4 == 0:
            self.tiny_slam.grid.display_cv(corrected_pose, goal=self.current_goal, traj= self.traj)  # display the map with the robot pose
        
        self.counter += 1
        # return self.control_tp1() # TP1 control
        return command  # control

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        
        # Reactive obstacle avoidance control
        #command, self.state, self.rotation_angle, self.range = reactive_obst_avoid(self.lidar(), self.state, self.rotation_angle, self.range)
        
        # Wall following control
        target_wall_dist = 30
        Kp = 0.008
        
        command , self.following_state = wall_following(
        self.lidar(),
        target_wall_dist,
        Kp,
        self.following_state)
        
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        goal = self.current_goal

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)
        
        if np.linalg.norm(pose[:2] - self.previous_pose[:2]) < 50:
            self.stuck_counter += 1
        else:
            self.stuck_counter = 0
            self.previous_pose = pose.copy()

        if self.stuck_counter > 1000:  # ~100 timesteps sans bouger
            self.current_goal = random_free_goal(self)
            self.exploration_counter += 1
            self.stuck_counter = 0
            print("Minimum local détecté, nouveau but tiré")
        
        if command["forward"] == 0.0 and command["rotation"] == 0.0:
            self.current_goal = random_free_goal(self)
            self.exploration_counter += 1
        return command

