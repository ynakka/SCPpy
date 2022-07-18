from time import time
import numpy as np

# map 
from src.Map.Map import Map
from src.Map.Obstacle import Obstacle


from src.SCP.SCP import SCP
# free flyer info
from free_flyer import free_flyer
from Robot import Robot

# plotting 
import matplotlib.pyplot as mpl
from plot_plan import plot_sphere

if __name__=="__main__":

    ## ------------ Setup Robot -----
    
    time_param = {}
    time_param["dt"] = 20
    time_param["num_tsteps"] = 20

    auto_free_flyer = Robot(time_param=time_param, dynamics= free_flyer)
    ## ------------ create Map ---------- 
    # Obstacle List
    # obstacle_list = []
    # radius = 0.3
    # location = np.array([0.0,0.0,0.0,0,0,0,0])
    # risk = 0.25
    # obstacle_list.append(Obstacle(radius=radius,\
    #         location=location,\
    #             risk = risk))

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

    map_setup = {}
    map_setup["lower_limit"] = np.array([-1.1,-1.1,-1.1]) 
    map_setup["upper_limit"] = np.array([1.3,1.3,1.1])



    robot_setup = {}
    robot_setup["initial_state"] = np.array([-1,-1,-1,\
        0,0,0,\
            0.,0.,0.,1,\
                0,0,0])
    robot_setup["initial_state_covariance"] = 0
    robot_setup["terminal_state"] = np.array([1.2,1.2,1.0,\
        0,0,0,\
        -0.5,0.5,-0.5,0.5,\
            0,0,0]) 
    robot_setup["terminal_state_covariance"] = 0

    Map1 = Map(obstacle_list=obstacle_list,\
        map_setup=map_setup,\
            robot_setup=robot_setup)

    # # ## -------- SCP ------------------
    initialization = {}
    initialization["valid"] = False
    scp_planner = SCP(Robot=auto_free_flyer,Map=Map1,initialization=initialization)
    scp_planner.scp()
    xscp = scp_planner.sol["state"]
    uscp = scp_planner.sol["control"]

    np.save('cluttered_xscp',xscp)
    np.save('cluttered_uscp',uscp)

    fig = mpl.figure()
    ax = mpl.axes(projection='3d')
    ax.view_init(30, 30)
    mpl.plot(xscp[:,0],xscp[:,1],xscp[:,2],label="SCP",linewidth=3)


    for i in range(len(obstacle_list)):
        plot_sphere(ax, obstacle_list[i].radius, obstacle_list[i].state)
    # mpl.plot(xscp[:,0],xscp[:,1])
    # ax1.plot(xgpc[:,0],xgpc[:,gpc_planner.num_gpcpoly])
    
    # Map1.plot_map(ax1)
    ax.set_xlabel('X (m)',fontsize=12)
    ax.set_ylabel('Y (m)',fontsize=12) 
    ax.set_zlabel('Z (m)',fontsize=12)
    # ax.yaxis._axinfo['label']['space_factor'] = 3.0
    # mpl.show()
    ax.grid(False)
    mpl.legend(fontsize=12)
    mpl.tight_layout()
    mpl.savefig('6dof_f2.pdf',dpi =300)