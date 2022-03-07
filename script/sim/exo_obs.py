import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.append("./script")

from utils import Kinematics, InverseKinematics

class EXO_OBS:
    """
    """
    def __init__(self,thigh=0.44,shank=0.565):
        """initial exo parameters

        Args:
            thigh (float): thigh link length
            shank (float): shank link length
        """
        self.thigh=thigh
        self.shank=shank
        
        # obstacle location and size
        self.obs_dist = 0.2 # assume obstacle locates in x-axis z = 0
        self.obs_height = 0.08
        self.obs_width = 0.08
        
        # set the position of stance ankle, which is used to calculate the exo joint position in world coordination
        self.ankle_offset_px = 0.0
        self.ankle_offset_pz = 0.0
        self.last_step_is_left = True 
        
        # foot trajectory
        self.traj = []
    
    def update_stace_leg(self,is_left_stance,lh,lk,rh,rk):
        """update the offset position when changing the stance leg

        Args:
            is_left_stance (bool): _description_
            lh (float): left hip angle (degree)
            lk (float): left knee angle
            rh (float): right hip angle
            rk (float): right knee angle
        """
        self.traj = [] # clear list
        
        l_ankle_px, l_ankle_pz = Kinematics(lh,lk,self.thigh,self.shank)
        r_ankle_px, r_ankle_pz = Kinematics(rh,rk,self.thigh,self.shank)
        
        if self.last_step_is_left != is_left_stance:
            self.ankle_offset_px += abs(l_ankle_px-r_ankle_px)

        self.last_step_is_left = is_left_stance
        
        
    def plot_exo(self,is_left_stance,lh,lk,rh,rk):
        
        # cal obstacle corner position
        obstacle_px = np.array([self.obs_dist,self.obs_dist,self.obs_dist+self.obs_width,self.obs_dist+self.obs_width,self.obs_dist])
        obstacle_pz = np.array([0,self.obs_height,self.obs_height,0,0])
        
        # cal exo joint position
        l_foot_px, l_foot_pz = Kinematics(lh,lk,self.thigh,self.shank)
        r_foot_px, r_foot_pz = Kinematics(rh,rk,self.thigh,self.shank)
        
        if is_left_stance:
            l_ankle_z = 0
            l_ankle_x = self.ankle_offset_px
            hip_z = l_ankle_z - l_foot_pz
            hip_x = l_ankle_x - l_foot_px
            l_knee_z = hip_z + self.thigh * np.sin(lh/180*np.pi- np.pi/2)
            l_knee_x = hip_x + self.thigh * np.cos(lh/180*np.pi - np.pi/2)
            
            r_knee_z = hip_z + self.thigh * np.sin(rh/180*np.pi- np.pi/2)
            r_knee_x = hip_x + self.thigh * np.cos(rh/180*np.pi - np.pi/2)
            r_ankle_z = hip_z + r_foot_pz
            r_ankle_x = hip_x + r_foot_px
            
            self.traj.append([r_ankle_x,r_ankle_z])
        else:
            r_ankle_z = 0
            r_ankle_x = self.ankle_offset_px
            hip_z = r_ankle_z - r_foot_pz
            hip_x = r_ankle_x - r_foot_px
            r_knee_z = hip_z + self.thigh * np.sin(rh/180*np.pi- np.pi/2)
            r_knee_x = hip_x + self.thigh * np.cos(rh/180*np.pi - np.pi/2)
            
            l_knee_z = hip_z + self.thigh * np.sin(lh/180*np.pi- np.pi/2)
            l_knee_x = hip_x + self.thigh * np.cos(lh/180*np.pi - np.pi/2)
            l_ankle_z = hip_z + l_foot_pz
            l_ankle_x = hip_x + l_foot_px
            
            self.traj.append([l_ankle_x,l_ankle_z])
            
        left_leg_px = np.array([hip_x,l_knee_x,l_ankle_x])
        left_leg_pz = np.array([hip_z,l_knee_z,l_ankle_z])
        right_leg_px = np.array([hip_x,r_knee_x,r_ankle_x])
        right_leg_pz = np.array([hip_z,r_knee_z,r_ankle_z])
        
        sw_traj = np.array(self.traj)
        
        # plot
        plt.clf()
        plt.plot(obstacle_px,obstacle_pz,'r',lw=3.0)
        plt.plot(left_leg_px,left_leg_pz,'b-o',lw=3.0)
        plt.plot(right_leg_px,right_leg_pz,'r-*',lw=3.0)
        
        plt.plot(sw_traj[:,0],sw_traj[:,1],'g',lw=2.0)
        
        plt.xlim([-0.5, 1.2])
        plt.ylim([-0.05, 1.2])
        
        plt.ioff()
        plt.pause(0.02)
       
        # plt.show()