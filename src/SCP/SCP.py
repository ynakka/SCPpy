import cvxpy as cp
import numpy as np

import jax.numpy as onp
# plotting 
import matplotlib.pyplot as mpl

class SCP():
    def __init__(self,Robot,Map,initialization):
        
        self.Robot = Robot
        self.Map = Map 

        self.dyn = Robot.dyn
        self.A = Robot.A
        self.B = Robot.B

        self.dt = Robot.dt

        self.num_states = Robot.num_states
        self.num_control = Robot.num_control

        self.u_max = Robot.u_max
        self.u_min = Robot.u_min

        # self.lower_limit = Robot.lower_limit
        # self.upper_limit = Robot.upper_limit

        self.radius = Robot.radius

        self.initial_state = Map.initial_state
        self.terminal_state = Map.terminal_state
        self.obstacle_list = Map.obstacle_list
        self.num_obstacles = len(self.obstacle_list)

        self.state_min = np.concatenate((Map.lower_limit,Robot.lower_limit))
        self.state_max = np.concatenate((Map.upper_limit,Robot.upper_limit))


        self.initialization = initialization
        self.nominal_trajectory = {}
        self.nominal_trajectory["valid"] = self.initialization["valid"]
        if self.nominal_trajectory["valid"] == True:            
            self.nominal_trajectory["state"] = self.initialization["state"]
            self.nominal_trajectory["control"] = self.initialization["control"]
            self.num_tsteps = self.nominal_trajectory["state"].shape[0]
        # print(self.num_tsteps)
        else:
            self.num_tsteps = self.Robot.num_tsteps
            self.nominal_trajectory["state"] = np.linspace(self.initial_state,self.terminal_state,self.num_tsteps)
            self.nominal_trajectory["control"] = 0.01*np.ones([int(self.num_tsteps-1),self.num_control])

        self.sol= {}
        self.sol["state"] = []
        self.sol["control"] = []
        self.sol["dt"] = self.Robot.dt
        
        
        self.trust = {"x":100,"u":100}
        self.scp_param = {}
        self.scp_param['error_tolerance'] = 0.001
        self.scp_param['alpha'] = 1.2 
        self.scp_param['beta'] = 0.5
        self.scp_param['iter_max'] = 10


    def scp(self):
        error = 1
        iterat = 1
        xprev = self.nominal_trajectory["state"]
        uprev = self.nominal_trajectory["control"]
        while iterat <= self.scp_param["iter_max"] and error >= self.scp_param["error_tolerance"]:
            # print('xprev:',xprev[:,1])
            result = self.convex_program(xprev,uprev)
            if result == 'INCREASE_TRUST':
                self.trust['x'] = self.trust['x']*self.scp_param["alpha"]
                self.trust['u'] = self.trust['u']*self.scp_param["alpha"]
                print('Exception Occured, Increasing Trust to :',self.trust['x'])
                continue
            else:
                x_sol = result[0]
                u_sol = result[1]
                iterat = iterat +1
                self.trust['x'] = self.trust['x']*self.scp_param["beta"]
                self.trust['u'] = self.trust['u']*self.scp_param["beta"]
                error =  np.amax(np.linalg.norm(x_sol-xprev,axis=1)) 
                print('iter :',iterat,'error :', error)
                # update previous and nominal trajectory
                xprev = x_sol
                uprev = u_sol
                # print('x shape:',x_sol.shape[0])
                # print('x:',x_sol[:,0])
                # print('y:',x_sol[:,1])

        self.sol["state"] = xprev
        self.sol["control"] = uprev
        print("SCP Converged!, Access solution using self.sol")
        return 1 

    def collision_constraint_3d(self,radius,obstacle_state,Xprev):
        mean = Xprev.reshape((self.num_states,1))
        G = np.zeros((self.num_states,self.num_states)) # matrix to  pulling out position 
        G[0,0] = 1
        G[1,1] = 1
        G[2,2] = 1
        mean_position = (np.mat(G)*np.mat(mean)).reshape((self.num_states)) 
        #print(mean_position)
        collision_dist = obstacle_state.reshape((self.num_states)) - mean_position
        #print(obstacle_state)
        #print(mean_position)
        a = collision_dist.reshape((self.num_states,1))
        #print(a)
        b =  np.mat(-a.reshape((1,self.num_states)))*np.mat(obstacle_state.reshape((self.num_states,1))) \
            + radius*np.linalg.norm(collision_dist,2)
        #print(b)
        return np.array(a,dtype=float), np.array(b,dtype=float)


    def convex_program(self,xprev,uprev):
        X = cp.Variable((self.num_tsteps,self.num_states))
        U = cp.Variable((self.num_tsteps-1,self.num_control))
        terminal_slack = cp.Variable(1)
        ## Initial Condition 
        constraint = []
        constraint.append(X[0]==self.initial_state)
        # print('init',self.initial_state)
        ## Terminal Condition
        constraint.append(cp.norm(X[self.num_tsteps-1] - self.terminal_state)<=terminal_slack)
        # print('term',self.terminal_state)
        ## Obstacles
        for t in range(self.num_tsteps-1):
            for jj in range(self.num_obstacles):
                # print(self.num_obstacles)
                a1, b1 = self.collision_constraint_3d(radius=self.obstacle_list[jj].radius + self.radius,\
                    obstacle_state=self.obstacle_list[jj].gpc_state,Xprev=xprev[t])
                # a1, b1 = self.Map.obstacle_list[jj].stationary_collision_constraint(xprev[jj])
                constraint.append(a1.T@X[t]+np.reshape(b1,1)<=0)

        ## Dynamics 
        for t in range(self.num_tsteps-1):
            x_dummy = onp.array(xprev[t],dtype=np.float32)
            u_dummy = onp.array(uprev[t],dtype=np.float32)
            Aa = self.A(x_dummy,u_dummy,self.dt)
            Ba = self.B(x_dummy,u_dummy,self.dt)
            Ca =  self.dyn(x_dummy,u_dummy,self.dt)- Aa@x_dummy- Ba@u_dummy
            constraint.append(X[t+1]== X[t]+ Aa@X[t] + Ba@U[t] + Ca) #

        # constraint on angular rotation 
        for t in range(self.num_tsteps-1):
            for i in range(self.num_states):
                constraint.append(X[t,i]<=self.state_max[i])
                constraint.append(X[t,i]>=self.state_min[i])
        
        # State trust region 
        for t in range(self.num_tsteps):
            constraint.append(cp.norm(X[t,:] - xprev[t,:] ) <= self.trust['x'])

        # Control trust region 
        for t in range(self.num_tsteps-1):
            constraint.append(cp.norm(U[t,:] - uprev[t,:] ) <= self.trust['u'])


        # control constraints
        for t in range(self.num_tsteps-1):
            constraint.append(U[t]<=self.u_max)
            constraint.append(U[t]>=self.u_min)
            
        cost = 1e3*terminal_slack
        # cost += cp.sum_squares(U-uprev)*self.dt
        for t in range(self.num_tsteps-1):
            cost += cp.quad_form(U[t],np.eye(self.num_control))*self.dt #cp.norm2(U[t])*dt 
        
        # terminal state cost 
        # cost += 1e3*cp.quad_form(X[self.num_tsteps-1]-self.terminal_state,np.eye(self.num_states))
        problem = cp.Problem(cp.Minimize(cost),constraint)

        try:
            result = problem.solve(solver=cp.ECOS)
        except:
            return 'INCREASE_TRUST'
        # Xv = np.around(X.value,decimals = 5)
        # Uv = np.around(U.value,decimals = 5)
        return [X.value,U.value]
    