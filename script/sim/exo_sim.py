import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("./script")

from utils import Kinematics, InverseKinematics

class EXO_SIM:
    """
    """
    def __init__(self,obs_d=0.1,obs_h=0.1,obs_w=0.1,thigh=0.44,shank=0.565):
        """initial exo parameters

        Args:
            thigh (float): thigh link length
            shank (float): shank link length
        """
        self.thigh=thigh
        self.shank=shank
        
        # obstacle location and size
        self.obs_dist = obs_d # assume obstacle locates in x-axis z = 0
        self.obs_height = obs_h
        self.obs_width = obs_w
        
        # set the position of stance ankle, which is used to calculate the exo joint position in world coordination
        self.ankle_offset_px = 0.0
        self.ankle_offset_pz = 0.0
        self.last_step_is_left = True 
        
        # exo joint status and left foot position
        self.lh = 0.0
        self.lk = 0.0
        self.rh = 0.0
        self.rk = 0.0
        self.lf_px = 0.0
        self.lf_pz = 0.0
        
        # swing foot trajectory
        self.traj = []
    
    def update_stace_leg(self,is_left_stance):
        """update the offset position when changing the stance leg

        Args:
            is_left_stance (bool): _description_
            lh (float): left hip angle (degree)
            lk (float): left knee angle
            rh (float): right hip angle
            rk (float): right knee angle
        """
        self.traj = [] # clear list
        
        l_ankle_px, l_ankle_pz = Kinematics(self.lh,self.lk,self.thigh,self.shank)
        r_ankle_px, r_ankle_pz = Kinematics(self.rh,self.rk,self.thigh,self.shank)
        
        if self.last_step_is_left != is_left_stance:
            self.ankle_offset_px += abs(l_ankle_px-r_ankle_px)

        self.last_step_is_left = is_left_stance
        
    def get_dist_obstacle(self):
        """return the distance between stance foot and obstacle

        Returns:
            float: distance between stance foot and obstacle
        """
        return self.obs_dist-self.ankle_offset_px
   
    def plot_one_step(self,st_h,st_k,sw_h,sw_k,plot_mode):
        """_summary_

        Args:
            is_left_stance (bool): 
            st_h (_type_): raw stance hip joint angle
            st_k (_type_): raw stance knee joint angle
            sw_h (_type_): raw swing hip joint angle
            sw_k (_type_): raw swing knee joint angle
        """
        num = st_h.shape[0]
        for i in range(num//5):
            j = i * 5
            self.plot_exo(st_h[j],st_k[j],sw_h[j],sw_k[j],plot_mode)      
    
    def plot_exo(self,lh,lk,rh,rk,plot_mode):
        
        # update exo joint angles
        self.lh = lh
        self.lk = lk
        self.rh = rh
        self.rk = rk
        
        # plot exo, obstacle, slope surface
        plt.clf()
        
        if plot_mode == "levelground":
            self.plot_level_walking()
            plt.legend(('stance leg','swing leg','swing foot traj'))
        elif plot_mode ==  "obstacle":
            self.plot_level_walking()
            self.plot_obstacle()
            plt.legend(('stance leg','swing leg','swing foot traj','obstacle'))
        elif plot_mode ==  "slope": 
            self.plot_level_walking()
            self.plot_slope()
            plt.legend(('stance leg','swing leg','swing foot traj','slope'))

        plt.xlim([-1.0, 1.2])
        plt.ylim([-0.25, 1.2])
        
        plt.ioff()
        plt.pause(0.000001)
        
    def plot_level_walking(self):
        # cal exo joint position in local frame whose origin is hip joint
        l_foot_px, l_foot_pz = Kinematics(self.lh,self.lk,self.thigh,self.shank)
        r_foot_px, r_foot_pz = Kinematics(self.rh,self.rk,self.thigh,self.shank)
        # transfer the joint position in the hip coordinate to the world coordinate
        if self.last_step_is_left:
            l_ankle_z = 0
            l_ankle_x = self.ankle_offset_px
            hip_z = l_ankle_z - l_foot_pz
            hip_x = l_ankle_x - l_foot_px
            l_knee_z = hip_z + self.thigh * np.sin(self.lh/180*np.pi- np.pi/2)
            l_knee_x = hip_x + self.thigh * np.cos(self.lh/180*np.pi - np.pi/2)
            
            r_knee_z = hip_z + self.thigh * np.sin(self.rh/180*np.pi- np.pi/2)
            r_knee_x = hip_x + self.thigh * np.cos(self.rh/180*np.pi - np.pi/2)
            r_ankle_z = hip_z + r_foot_pz
            r_ankle_x = hip_x + r_foot_px
            
            # add trajectory point to plot
            self.traj.append([r_ankle_x,r_ankle_z])
        else:
            r_ankle_z = 0
            r_ankle_x = self.ankle_offset_px
            hip_z = r_ankle_z - r_foot_pz
            hip_x = r_ankle_x - r_foot_px
            r_knee_z = hip_z + self.thigh * np.sin(self.rh/180*np.pi- np.pi/2)
            r_knee_x = hip_x + self.thigh * np.cos(self.rh/180*np.pi - np.pi/2)
            
            l_knee_z = hip_z + self.thigh * np.sin(self.lh/180*np.pi- np.pi/2)
            l_knee_x = hip_x + self.thigh * np.cos(self.lh/180*np.pi - np.pi/2)
            l_ankle_z = hip_z + l_foot_pz
            l_ankle_x = hip_x + l_foot_px
            
            self.traj.append([l_ankle_x,l_ankle_z])
        
        # update exo joint and left foot position
        self.lf_px = l_knee_x
        self.lf_pz = l_knee_z
        
        # plot obstacle, exo leg, and swing foot trajectory
        left_leg_px = np.array([hip_x,l_knee_x,l_ankle_x])
        left_leg_pz = np.array([hip_z,l_knee_z,l_ankle_z])
        right_leg_px = np.array([hip_x,r_knee_x,r_ankle_x])
        right_leg_pz = np.array([hip_z,r_knee_z,r_ankle_z])
        sw_traj = np.array(self.traj)
        
        plt.plot(left_leg_px,left_leg_pz,'b-o',lw=3.0)
        plt.plot(right_leg_px,right_leg_pz,'r-*',lw=3.0)
        plt.plot(sw_traj[:,0],sw_traj[:,1],'g',lw=2.0)
        
    def plot_obstacle(self):
        # cal obstacle corner position
        obstacle_px = np.array([self.obs_dist,self.obs_dist,self.obs_dist+self.obs_width,self.obs_dist+self.obs_width,self.obs_dist])
        obstacle_pz = np.array([0,self.obs_height,self.obs_height,0,0])
        plt.plot(obstacle_px,obstacle_pz,'r',lw=3.0)
        
    def plot_slope(self):
        gradient = 10       