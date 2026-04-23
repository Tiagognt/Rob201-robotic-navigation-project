"""
Robot controller definition
Complete controller including SLAM, planning, path following
"""
import numpy as np

from place_bot.simulation.robot.robot_abstract import RobotAbstract
from place_bot.simulation.robot.odometer import OdometerParams
from place_bot.simulation.ray_sensors.lidar import LidarParams

from tiny_slam import TinySlam

from control import potential_field_control, reactive_obst_avoid, wall_following
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
        
        # attributes for TP1 bonus controler
        self._wall_following_state = "search"  # state for wall following behavior
        self.rotation_side = "left"  # side to turn when following a wall

        # Init SLAM object
        # Here we cheat to get an occupancy grid size that's not too large, by using the
        # robot's starting position and the maximum map size that we shouldn't know.
        size_area = (1400, 1000)
        robot_position = (439.0, 195)
        self.occupancy_grid = OccupancyGrid(x_min=-(size_area[0] / 2 + robot_position[0]),
                                            x_max=size_area[0] / 2 - robot_position[0],
                                            y_min=-(size_area[1] / 2 + robot_position[1]),
                                            y_max=size_area[1] / 2 - robot_position[1],
                                            resolution=2)

        self.tiny_slam = TinySlam(self.occupancy_grid)
        self.planner = Planner(self.occupancy_grid)

        # storage for pose after localization
        self.corrected_pose = np.array([0, 0, 0])
        

    def control(self):
        """
        Main control function executed at each time step
        """
        
        # Section of the TP3 
        odom_pose = self.odometer_values()
        lidar = self.lidar()
        
        """
        # the TODO section for TP4

        self.tiny_slam.update_map(self.lidar(), odom_pose)
        if self.counter % 10 == 0:  
            self.tiny_slam.grid.display_cv(odom_pose, traj=None, goal=None)  # display the map with the robot pose
        
        """    # Mise à jour de la carte seulement si le score est bon

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
        
        if self.counter % 4 == 0:
            self.tiny_slam.grid.display_cv(corrected_pose)


        
        self.counter += 1
        return self.control_tp1() # TP1 control
        # return self.control_tp2()  # TP2 control

    def control_tp1(self):
        """
        Control function for TP1
        Control funtion with minimal random motion
        """
        
        # Reactive obstacle avoidance control
        command, self.state, self.rotation_angle, self.range = reactive_obst_avoid(self.lidar(), self.state, self.rotation_angle, self.range)
        
        # Wall following control
        
        return command

    def control_tp2(self):
        """
        Control function for TP2
        Main control function with full SLAM, random exploration and path planning
        """
        pose = self.odometer_values()
        goal = [-150,-430,0]

        # Compute new command speed to perform obstacle avoidance
        command = potential_field_control(self.lidar(), pose, goal)

        return command

