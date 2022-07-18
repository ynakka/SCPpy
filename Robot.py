import numpy as np
from jax import jit, jacfwd

class Robot():
    def __init__(self,time_param,dynamics) -> None:
        self.dyn = jit(dynamics)
        self.radius = 0.0
        self.num_states = 13
        self.num_control = 6

        self.dt = time_param["dt"]
        self.num_tsteps = time_param["num_tsteps"]

        self.upper_limit = np.array([0.1,0.1,0.1,\
               100.,100.,100.,100.,\
                    0.1,0.1,0.1])
        self.lower_limit = np.negative(self.upper_limit)

        self.u_max = 5*np.array([1,1,1,\
              0.3,0.3,0.3])
        self.u_min = np.negative(self.u_max)

        self.A = jit(jacfwd(self.dyn,0))
        self.B = jit(jacfwd(self.dyn,1))

        pass