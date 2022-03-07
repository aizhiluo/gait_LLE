import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import csv

import sys
sys.path.append("./script")

from copy import deepcopy
from dmp import DMP
from utils import Kinematics, InverseKinematics, JacobianMatrix


# obstacle location and size
obstacle_dist = 0.2 # assume obstacle locates in x-axis z = 0
obstacle_height = 0.08
obstacle_width = 0.08

shank_length = 0.565
thigh_length = 0.44
foot_length = 0.0
ankle_angle = 0.0

in_joint_space = False

def PotentialForceFile(px,pz):
    
    if type(px) == np.ndarray:
        force_x = np.zeros(px.shape[0])
        force_z = np.zeros(px.shape[0])
    else:
        force_x = 0.0
        force_z = 0.0

    only_one_point = True

    alpha = 1.0
    beta = 63.25
    k = 10.0
    corner_x = np.array([obstacle_dist,obstacle_dist,obstacle_dist+obstacle_width,obstacle_dist+obstacle_width])
    corner_z = np.array([0.0,obstacle_height,obstacle_height,0.0])
    center_x = np.mean(corner_x)
    center_z = np.mean(corner_z)
    
    # potential fields repulsive force formula parameters
    radius = max(obstacle_height,obstacle_width) / np.sqrt(2)
    D = 0.05 # tolent space
    k = alpha / (radius + D)
    beta = np.sqrt(k/(radius * D))
    
    if only_one_point is True:
        # only considering the center point of the obstacle
        dist = (px-center_x)**2 + (pz-center_z)**2
        force = alpha/(k+dist*beta*beta)
        force_x = force * (px-center_x)/np.sqrt(dist) * 0.0
        force_z = force * (pz-center_z)/np.sqrt(dist)
        
    else:
        # consider four corner of the obstacle
        for i in range(4):
            dist = (px-corner_x[i])**2 + (pz-corner_z[i])**2
            force = alpha/(k+dist*beta*beta)
            force_x += force * (px-corner_x[i])/np.sqrt(dist) * 0.0
            force_z += force * (pz-corner_z[i])/np.sqrt(dist)
    
    return force_x, (force_z + abs(force_z))/2.0
        
if __name__ == "__main__":
    
    path = "./data/joint_angle/"
    file_name = "SN_all_subj_obstacle_first"

    file_path = path + file_name + ".txt"
    txt_file = open(file_path, 'r')
    lines = txt_file.readlines()[2:]

    st_hip_list = []
    st_knee_list = []
    sw_hip_list = []
    sw_knee_list = []
    for l in lines:
        str = l.replace('\n', '').split(",")
        st_hip_list.append(float(str[0]))
        sw_hip_list.append(float(str[1]))
        st_knee_list.append(float(str[2]))
        sw_knee_list.append(float(str[3]))
        
    hip_angle = np.array([st_hip_list,sw_hip_list])
    knee_angle = np.array([st_knee_list,sw_knee_list])
    foot_px,foot_pz = Kinematics(hip_angle,knee_angle,thigh_length,shank_length)
    
    # relative distance between swing and stance foot
    px = foot_px[1,:] - foot_px[0,:]
    pz = foot_pz[1,:] - foot_pz[0,:]
    
    len = hip_angle.shape[1]
    post_sw_px = np.zeros(len)
    post_sw_pz = np.zeros(len)
    post_sw_hip = np.zeros(len)
    post_sw_knee = np.zeros(len)
    post_px = np.zeros(len)
    post_pz = np.zeros(len)
    force_x = np.zeros(len)
    force_z = np.zeros(len)
    
    if in_joint_space is False:
        # adjust endeffector position based on obatacle position
        force_x,force_z = PotentialForceFile(px,pz)
        post_sw_px = foot_px[1,:] + force_x
        post_sw_pz = foot_pz[1,:] + force_z
        post_px = post_sw_px - foot_px[0,:]
        post_pz = post_sw_pz - foot_pz[0,:]
        post_sw_hip,post_sw_knee = InverseKinematics(np.array([post_sw_px,post_sw_pz]),thigh_length,shank_length)
        
    else:
        #######   DMP fitting   ######
        y_des = np.array([st_hip_list, sw_hip_list, st_knee_list, sw_knee_list])
        y_des_time = np.arange(0, y_des.shape[1])*0.001
        
        ay = np.ones(4) * 50
        by = ay / 4
        dmp_gait = DMP(y_des, y_des_time, ay=ay, by=by, n_bfs=450, dt=0.001, isz=True)
        _, dmp_time_left_swing = dmp_gait.imitate_path()
        
        ########  Test DMP   #############
        traj_num = y_des.shape[0]
        num_steps = y_des.shape[1]
        track = np.zeros((traj_num, num_steps))
        dmp_time = np.arange(num_steps) * 0.001
        
        # the goal_offset, scale, and initial position for the generated trajectory
        goal_offset = np.array([0,0,0,0])
        new_scale = np.ones(num_steps)
        y = y_des[:,0]
        dy = np.zeros(traj_num)
        force = np.zeros(traj_num)
        
        # Generate target trajectory using DMP step by step
        for i in range(num_steps):            
            gait_phase = float(i) / num_steps
            y, dy, ddy = dmp_gait.step_real(gait_phase, y, dy, scale=new_scale, goal_offset=goal_offset, tau=1.0)
            y += force
            
            track[:, i] = deepcopy(y)
            
            sw_hip_angle = y[1]
            sw_knee_angle = y[3]
            post_sw_hip[i] = y[1]
            post_sw_knee[i] = y[3]
             
            # distance between both legs
            tmp_px,tmp_pz = Kinematics(y[0:2],y[2:4],thigh_length,shank_length)
            post_px[i] = tmp_px[1] - tmp_px[0]
            post_pz[i] = tmp_pz[1] - tmp_pz[0]
            
            # potential force
            tmp_force_x,tmp_force_z = PotentialForceFile(post_px[i],post_pz[i])
            force_x[i] = tmp_force_x
            force_z[i] = tmp_force_z
            
            # transfer force to joint torque by JacobianMatrix
            JacoM = JacobianMatrix(sw_hip_angle,sw_knee_angle,thigh_length,shank_length)
            tmp_torque_hip = JacoM[0,0] * tmp_force_x + JacoM[1,0] * tmp_force_z
            tmp_torque_knee = JacoM[0,1] * tmp_force_x + JacoM[1,1] * tmp_force_z
            force = np.array([0,tmp_torque_hip*40,0,tmp_torque_knee*20]) * 1.0
            
            force_x[i] = tmp_torque_hip
            force_z[i] = tmp_torque_knee
        
        dmp_st_hip = track[0,:]
        dmp_sw_hip = track[1,:]
        dmp_st_knee = track[2,:]
        dmp_sw_knee = track[3,:]
        
        # Plot original and generated trajectories
        plt.figure(3)
        plt.subplot(221)
        plt.plot(y_des_time, st_hip_list)
        plt.plot(dmp_time, dmp_st_hip)
        plt.legend(('st hip','dmp st hip'))
        plt.subplot(223)
        plt.plot(y_des_time, st_knee_list, label='st_knee')
        plt.plot(dmp_time, dmp_st_knee, label='dmp st_knee')
        plt.legend(('st knee','dmp st knee'))
        plt.subplot(222)
        plt.plot(y_des_time, sw_hip_list, label='sw_hip')
        plt.plot(dmp_time, dmp_sw_hip, label='dmp sw_hip')
        plt.legend(('sw hip','dmp sw hip'))
        plt.subplot(224)
        plt.plot(y_des_time, sw_knee_list, label='sw_knee')
        plt.plot(dmp_time, dmp_sw_knee, label='dmp sw_knee')
        plt.legend(('sw knee','dmp sw knee'))
    
    
    # plot joint angle axs[0,0].
    #fig, axs = plt.subplots(2,2,sharey="row")
    plt.figure(1)
    plt.subplot(221)
    plt.plot(hip_angle[0,:],'r')
    plt.plot(knee_angle[0,:],'b')
    plt.legend(('st hip','st knee'))
    
    plt.subplot(223)
    plt.plot(hip_angle[0,:],'r')
    plt.plot(knee_angle[0,:],'b')
    plt.legend(('post st hip','post st knee'))
    
    plt.subplot(222)
    plt.plot(foot_px[0,:],foot_pz[0,:],'b',lw=2.0) # end effector trajectory
    plt.plot(foot_px[0,0],foot_pz[0,0],'bo',lw=5.0)
    plt.plot(foot_px[1,:],foot_pz[1,:],'r',lw=2.0) # end effector trajectory
    plt.plot(foot_px[0,1],foot_pz[0,1],'ro',lw=5.0)
    plt.legend(('st foot',' ','sw foot',' '))
  
    # plot end effector position  
    obstacle_px = np.array([obstacle_dist,obstacle_dist,obstacle_dist+obstacle_width,obstacle_dist+obstacle_width,obstacle_dist])
    obstacle_pz = np.array([0,obstacle_height,obstacle_height,0,0])
    
    plt.figure(2)
    
    plt.subplot(221)
    plt.plot(hip_angle[1,:],'r')
    plt.plot(knee_angle[1,:],'b')
    plt.legend(('sw hip','sw knee'))
    
    plt.subplot(222)
    plt.plot(post_sw_hip,'r')
    plt.plot(post_sw_knee,'b')
    plt.legend(('post sw hip','post sw knee'))
    
    plt.subplot(223)
    plt.plot(px,pz,'c-',lw=2.0) # end effector trajectory
    plt.plot(post_px,post_pz,'b--',lw=2.0)
    plt.plot(obstacle_px,obstacle_pz,'r',lw=3.0)
    plt.legend(('original end effector','post end effector','obstacle'))
    
    plt.subplot(224)
    plt.plot(force_x)
    plt.plot(force_z)
    plt.legend(('force x','force z'))
    
    plt.show()