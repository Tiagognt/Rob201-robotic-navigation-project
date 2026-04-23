""" A simple robotics navigation code including SLAM, exploration, planning"""

import cv2
import numpy as np
from occupancy_grid import OccupancyGrid


class TinySlam:
    """Simple occupancy grid SLAM"""

    def __init__(self, occupancy_grid: OccupancyGrid):
        self.grid = occupancy_grid

        # Origin of the odom frame in the map frame
        self.odom_pose_ref = np.array([0, 0, 0])

    def _score(self, lidar, pose):
        """
        Computes the sum of log probabilities of laser end points in the map
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, position of the robot to evaluate, in world coordinates
        """
        # TODO for TP4

        lidar_dist = lidar.get_sensor_values()
        lidar_angles = lidar.get_ray_angles()
        
        valid_dist = lidar_dist <= lidar.max_range 
        lidar_dist = lidar_dist[valid_dist]
        lidar_angles = lidar_angles[valid_dist]
        
        lidar_x = pose[0] + lidar_dist * np.cos(pose[2] + lidar_angles)
        lidar_y = pose[1] + lidar_dist * np.sin(pose[2] + lidar_angles)
        
        map_x, map_y = self.grid.conv_world_to_map(lidar_x, lidar_y)
        
        in_map = (map_x >= 0) & (map_x < self.grid.x_max_map) & (map_y >= 0) & (map_y < self.grid.y_max_map)
        
        map_x = map_x[in_map]
        map_y = map_y[in_map]
        
        score = np.sum(self.grid.occupancy_map[map_x, map_y])

        return score

    def get_corrected_pose(self, odom_pose, odom_pose_ref=None):
        """
        Compute corrected pose in map frame from raw odom pose + odom frame pose,
        either given as second param or using the ref from the object
        odom : raw odometry position
        odom_pose_ref : optional, origin of the odom frame if given,
                        use self.odom_pose_ref if not given
        """
        # TODO for TP4
        
        if odom_pose_ref is None:
            odom_pose_ref = self.odom_pose_ref
            
        d0 = np.sqrt(odom_pose[0]**2 + odom_pose[1]**2)
        alpha0 = np.arctan2(odom_pose[1], odom_pose[0])
        
        corrected_pose = np.zeros(3)

        corrected_pose[0] = odom_pose_ref[0] + d0 * np.cos(alpha0+ odom_pose_ref[2])
        corrected_pose[1] = odom_pose_ref[1] + d0 * np.sin(alpha0+ odom_pose_ref[2])
        corrected_pose[2] = odom_pose_ref[2] + odom_pose[2]

        return corrected_pose

    def localise(self, lidar, raw_odom_pose):
        """
        Compute the robot position wrt the map, and updates the odometry reference
        lidar : placebot object with lidar data
        odom : [x, y, theta] nparray, raw odometry position
        """
        # TODO for TP4

        sigma = np.array([0.5, 0.5, 0.02])  
        N_no_improve = 50                   

        current_pose = self.get_corrected_pose(raw_odom_pose)
        best_score = self._score(lidar, current_pose)
        best_ref = self.odom_pose_ref.copy()

        no_improve_count = 0
        while no_improve_count < N_no_improve:
            noise = np.random.normal(0, sigma)
            new_ref = best_ref + noise

            new_pose = self.get_corrected_pose(raw_odom_pose, new_ref)
            new_score = self._score(lidar, new_pose)

            if new_score > best_score:
                best_score = new_score
                best_ref = new_ref
                no_improve_count = 0 
            else:
                no_improve_count += 1

        self.odom_pose_ref = best_ref
        

        return best_score
        
    def update_map(self, lidar, pose,):
        """
        Bayesian map update with new observation
        lidar : placebot object with lidar data
        pose : [x, y, theta] nparray, corrected pose in world coordinates
        """
        # TODO for TP3
        
        
        laser_dist = lidar.get_sensor_values()
        laser_angles = lidar.get_ray_angles()
        
        # Compute the end points of the laser beams in world coordinates
        laser_x = pose[0] + laser_dist * np.cos(pose[2] + laser_angles)
        laser_y = pose[1] + laser_dist * np.sin(pose[2] + laser_angles)
        
        laser_x_free = pose[0] + (laser_dist - 15) * np.cos(pose[2] + laser_angles)
        laser_y_free = pose[1] + (laser_dist - 15) * np.sin(pose[2] + laser_angles)
        
        
        for i in range(len(laser_dist)):
            self.grid.add_value_along_line(pose[0], pose[1], laser_x_free[i], laser_y_free[i], -1)
            
        self.grid.add_map_points(laser_x, laser_y, 4)  # add occupied points to the map
        np.clip(self.grid.occupancy_map, -40, 40, out=self.grid.occupancy_map)  # clip values to prevent overflow
        
        
        
        

        
        
