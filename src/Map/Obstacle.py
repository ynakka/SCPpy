import numpy as np
from matplotlib.patches import Circle

class Obstacle:
    def __init__(self,radius,location,risk):
        """[summary]
        Args:
            dimenesion ([type]): '2' or '3' [description]
            radius ([type]): [description]
            location ([type]): [description]
            risk ([type]): [description]
        """
        self.radius = radius
        self.risk = risk
        self.state = np.array([location[0],location[1],location[2],0,0,0])
        self.gpc_state = np.zeros(13)
        self.gpc_state[0] = location[0]
        self.gpc_state[1] = location[1]
        self.gpc_state[2] = location[2]


