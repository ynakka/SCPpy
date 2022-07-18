'''
Description : Class that describes the Map for motion planning
'''
from Map.Obstacle import Obstacle


class Map:
    def __init__(self,obstacle_list,map_setup,robot_setup):
        """[summary]

        Args:
            state_init ([type]): [description]
            state_final ([type]): [description]
            obstacle_list ([type]): [description]
        """
        self.lower_limit = map_setup["lower_limit"]
        self.upper_limit = map_setup["upper_limit"]
        self.obstacle_list = obstacle_list
        self.dimension = '3'

        self.initial_state = robot_setup["initial_state"]
        self.initial_state_covariance = robot_setup["initial_state_covariance"]
        self.terminal_state = robot_setup["terminal_state"] 
        self.terminal_state_covariance = robot_setup["terminal_state_covariance"]
    

