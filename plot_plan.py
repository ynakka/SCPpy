
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as mpl
from src.Map.Obstacle import Obstacle
num_states = 13
num_uncertainty = 13+4
num_gpcpoly = 5
polynomial_degree = 1

##  ---- cluttered 3d space        
obstacle_list = []
radius = 0.3
location = np.array([-0.5,-0.5,0.5,0,0,0,0])
risk = 0.05 
obstacle_list.append(Obstacle(radius=radius,\
        location=location,\
            risk = risk))


radius = 0.2
location = np.array([0.5,0.3,0.5,0,0,0])
risk = 0.05 
obstacle_list.append(Obstacle(radius=radius,\
        location=location,\
            risk = risk))


radius = 0.3
location = np.array([0.5,-0.5,0.5,0,0,0])
risk = 0.05
obstacle_list.append(Obstacle(radius=radius,\
        location=location,\
            risk = risk))

radius = 0.15
location = np.array([-0.5,0.5,0.5,0,0,0])
risk = 0.05
obstacle_list.append(Obstacle(radius=radius,\
        location=location,\
            risk = risk))

radius = 0.1
location = np.array([-0.5,-0.5,-0.5,0,0,0,0])
risk = 0.05 
obstacle_list.append(Obstacle(radius=radius,\
        location=location,\
            risk = risk))


radius = 0.2
location = np.array([0.5,0.5,-0.5,0,0,0])
risk = 0.05 
obstacle_list.append(Obstacle(radius=radius,\
        location=location,\
            risk = risk))


radius = 0.3
location = np.array([0.5,-0.5,-0.5,0,0,0])
risk = 0.05
obstacle_list.append(Obstacle(radius=radius,\
        location=location,\
            risk = risk))

radius = 0.15
location = np.array([-0.5,0.5,-0.5,0,0,0])
risk = 0.05
obstacle_list.append(Obstacle(radius=radius,\
        location=location,\
            risk = risk))

radius = 0.15
location = np.array([-0.2,0.2,0.0,0,0,0])
risk = 0.05
obstacle_list.append(Obstacle(radius=radius,\
        location=location,\
            risk = risk))

def plot_sphere(ax,radius,location):
    u, v = np.mgrid[0:2 * np.pi:30j, 0:np.pi:20j]
    x = location[0] + radius*np.cos(u) * np.sin(v)
    y = location[1] + radius*np.sin(u) * np.sin(v)
    z = location[2] + radius*np.cos(v)
    return ax.plot_surface(x, y, z, color='r',alpha = 0.3)


if __name__=="__main__":
    print("")
