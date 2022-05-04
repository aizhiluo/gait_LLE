from sim import EXO_SIM
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import imageio
import os

from copy import deepcopy
from dmp import DMP
from utils import CorrectHipForStepLength, SlopeModelNoAnkleConstrain, SlopeModelWithAnkleConstrain, Kinematics, InverseKinematics, JacobianMatrix

def read_gait_data_txt(file_path):
    
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
    txt_file.close()
    
    st_hip = np.array(st_hip_list)
    st_knee = np.array(st_knee_list)
    sw_hip = np.array(sw_hip_list)
    sw_knee = np.array(sw_knee_list)
    
    return st_hip, sw_hip, st_knee, sw_knee
   
def dmp_gait_generation(dmp_gait,num_steps,y0,new_scale=None,goal_offset=None,forces=None):
    
    traj_num = y0.shape[0]
    track = np.zeros((traj_num, num_steps))
    
    tau = (dmp_gait.timesteps+1) / num_steps
    track_time = np.arange(num_steps) * dmp_gait.dt * tau
    
    # the goal_offset, scale, and initial position for the generated trajectory
    if new_scale is None:
        new_scale = np.ones(traj_num)
    if goal_offset is None:
        goal_offset = np.zeros(traj_num)
    if forces is None:
        forces = np.zeros((traj_num, num_steps))
        
    y = y0
    dy = np.zeros(traj_num)
     # Generate target trajectory using DMP step by step
    for i in range(num_steps):            
        gait_phase = float(i) / num_steps
        y, dy, ddy = dmp_gait.step_real(gait_phase,y,dy,scale=new_scale,goal_offset=goal_offset,tau=tau,extra_force=forces[:,i])
        track[:, i] = deepcopy(y)
              
    # dmp_st_hip = track[0,:]
    # dmp_sw_hip = track[1,:]
    # dmp_st_knee = track[2,:]
    # dmp_sw_knee = track[3,:]
    
    return track, track_time


# load gait data
path = "D:/MyFolder/code/gait_ws/gait_LLE/data/joint_angle/"
file_name = "CH_walking" #"SN_all_subj_obstacle_first" #
file_path = path + file_name + ".txt"
raw_st_hip, raw_sw_hip, raw_st_knee, raw_sw_knee=read_gait_data_txt(file_path)
num = raw_st_hip.shape[0]

file_name = "CH_upslope"
file_path = path + file_name + ".txt"
upslope_st_hip, upslope_sw_hip, upslope_st_knee, upslope_sw_knee=read_gait_data_txt(file_path)

file_name = "CH_downslope"
file_path = path + file_name + ".txt"
downslope_st_hip, downslope_sw_hip, downslope_st_knee, downslope_sw_knee=read_gait_data_txt(file_path)

# create and train DMP module
y_des = np.array([raw_st_hip, raw_sw_hip, raw_st_knee, raw_sw_knee])
y_des_time = np.arange(0, y_des.shape[1])*0.001
dmp_gait = DMP(y_des, y_des_time,n_bfs=250, dt=0.001, isz=True)
_, _, = dmp_gait.imitate_path()

# create and initial exo simulation models
with_ankle_constrain = True
is_save_gif = True
terrain_type = "slope"

slope = 5.0
step_length = 0.54

thigh_length = 0.413 #0.51(LC)
shank_length = 0.373 #0.41(LC)
normal_step_len = 0.54

obstacle_dist = 0.15 # assume obstacle locates in x-axis z = 0
obstacle_width = 0.1
obstacle_height = 0.1
obstacle_px = np.array([obstacle_dist,obstacle_dist,obstacle_dist+obstacle_width,obstacle_dist+obstacle_width,obstacle_dist])
obstacle_pz = np.array([0,obstacle_height,obstacle_height,0,0])

exo = EXO_SIM(obstacle_dist,obstacle_height,obstacle_width,thigh_length,shank_length,slope=slope)
exo.update_stace_leg(True)

# different terrain simulation
if terrain_type == "slope":
        
    y = y_des[:,0] + [0,0,0,0]
    new_scale = np.ones(y.shape[0]) + [0,0,0,0]
    goal_offset = np.zeros(y.shape[0]) + [0,0,0,0]
    
    # the improved slope model with ankle constrain
    # first step
    target_angle1 = SlopeModelWithAnkleConstrain(y_des[:,-1],thigh_length,shank_length,slope) 
    # Hip dmp unit needs the extral scale to correct the affect from the goal offset
    goal_offset = target_angle1 - y_des[:,-1]
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    new_scale[3] = (1.0 - (target_angle1[3] - 8.0) / 65.0) * step_length / normal_step_len
    track1,track1_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)
    # second step
    y = np.array([track1[1,-1],track1[0,-1],track1[3,-1],track1[2,-1]])
    target_angle2 = SlopeModelWithAnkleConstrain(y_des[:,-1],thigh_length,shank_length,slope)
    goal_offset = target_angle2 - y_des[:,-1]
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    new_scale[3] = (1.0 - (target_angle2[3] - 8.0) / 65.0) * step_length / normal_step_len
    track2,track2_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)


    # slope model with fixed stance hip and knee angle
    # first step
    target_angle3 = SlopeModelNoAnkleConstrain(y_des[:,-1],thigh_length,shank_length,slope)
    goal_offset = target_angle3 - y_des[:,-1]
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    new_scale[3] = (1.0 - (target_angle3[3] - 8.0) / 65.0) * step_length / normal_step_len
    track3,track3_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)
    # second step
    y = np.array([track3[1,-1],track3[0,-1],track3[3,-1],track3[2,-1]])
    target_angle4 = SlopeModelNoAnkleConstrain(y_des[:,-1],thigh_length,shank_length,slope)
    goal_offset = target_angle4 - y_des[:,-1]
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    new_scale[3] = (1.0 - (target_angle4[3] - 8.0) / 65.0) * step_length / normal_step_len
    track4,track4_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)
    
    print("ending joint angle in walking: ",y_des[:,-1])
    print("slope model with fixed stance leg: ",target_angle4)
    print("slope model with ankle constrain: ",target_angle2)
    
    # plot first step in slope
    st_hip=track1[0,:]
    sw_hip=track1[1,:]
    st_knee=track1[2,:]
    sw_knee=track1[3,:]
    f, axs = plt.subplots(2,2)
    axs[0][0].plot(y_des_time, raw_st_hip, label='raw st_hip')
    axs[1][0].plot(y_des_time, raw_st_knee, label='raw st_knee')  
    axs[0][1].plot(y_des_time, raw_sw_hip, label='raw sw_hip')    
    axs[1][1].plot(y_des_time, raw_sw_knee, label='raw sw_knee')
    
    axs[0][0].plot(track1_time, st_hip, label='dmp st_hip')
    axs[1][0].plot(track1_time, st_knee, label='dmp st_knee')
    axs[0][1].plot(track1_time, sw_hip, label='dmp sw_hip')
    axs[1][1].plot(track1_time, sw_knee, label='dmp sw_knee')
    
    axs[0][0].plot(y_des_time, upslope_st_hip, label='raw slope st_hip')
    axs[1][0].plot(y_des_time, upslope_st_knee, label='raw slope st_knee')
    axs[0][1].plot(y_des_time, upslope_sw_hip, label='raw slope sw_hip')
    axs[1][1].plot(y_des_time, upslope_sw_knee, label='raw slope sw_knee')
    
    for ax in axs:
        for a in ax:
            a.legend()
            
    # plot second step in slope
    st_hip=track2[0,:]
    sw_hip=track2[1,:]
    st_knee=track2[2,:]
    sw_knee=track2[3,:]
    f1, axs1 = plt.subplots(2,2)
    axs1[0][0].plot(y_des_time, raw_st_hip, label='mocap walking st_hip')
    axs1[1][0].plot(y_des_time, raw_st_knee, label='mocap walking st_knee')
    axs1[0][1].plot(y_des_time, raw_sw_hip, label='mocap walking sw_hip')
    axs1[1][1].plot(y_des_time, raw_sw_knee, label='mocap walking sw_knee')
    
    axs1[0][0].plot(y_des_time, upslope_st_hip, label='mocap slope st_hip')
    axs1[1][0].plot(y_des_time, upslope_st_knee, label='mocap slope st_knee')
    axs1[0][1].plot(y_des_time, upslope_sw_hip, label='mocap slope sw_hip')
    axs1[1][1].plot(y_des_time, upslope_sw_knee, label='mocap slope sw_knee')
    
    axs1[0][0].plot(y_des_time, track4[0,:], label='slope model st_hip')
    axs1[1][0].plot(y_des_time, track4[2,:], label='slope model st_knee')
    axs1[0][1].plot(y_des_time, track4[1,:], label='slope model sw_hip')
    axs1[1][1].plot(y_des_time, track4[3,:], label='slope model sw_knee')
    
    axs1[0][0].plot(track2_time, st_hip, label='improved model st_hip')
    axs1[1][0].plot(track2_time, st_knee, label='improved model st_knee')
    axs1[0][1].plot(track2_time, sw_hip, label='improved model sw_hip')
    axs1[1][1].plot(track2_time, sw_knee, label='improved model sw_knee')
    
    for ax in axs1:
        for a in ax:
            a.legend()
    
    if is_save_gif is False:
    # plot EXO simulation animation
        plt.figure(3)
        if with_ankle_constrain is True:
            lh = track1[0,:]
            lk = track1[2,:]
            rh = track1[1,:]
            rk = track1[3,:]
        else:
            lh = track3[0,:]
            lk = track3[2,:]
            rh = track3[1,:]
            rk = track3[3,:]
        
        exo.plot_one_step(lh,lk,rh,rk,terrain_type)
        
        if with_ankle_constrain is True:
            lh = track2[1,:]
            lk = track2[3,:]
            rh = track2[0,:]
            rk = track2[2,:]
        else:
            lh = track4[1,:]
            lk = track4[3,:]
            rh = track4[0,:]
            rk = track4[2,:]
        
        exo.update_stace_leg(False)
        exo.plot_one_step(lh,lk,rh,rk,terrain_type)
        plt.show()
    else:
        # save to gif
        plt.figure(3)
        filename = 'fig.png'
        if with_ankle_constrain is True:
            gif_file_name = "5slope_model_with_fixed_stance_leg.gif"
        else:
            gif_file_name = "5slope_model_with_ankle_constrain.gif"
            
        with imageio.get_writer(gif_file_name, mode='I') as writer:
            if with_ankle_constrain is True:
                lh = track1[0,:]
                lk = track1[2,:]
                rh = track1[1,:]
                rk = track1[3,:]
            else:
                lh = track3[0,:]
                lk = track3[2,:]
                rh = track3[1,:]
                rk = track3[3,:]
                
            for i in range(num//5):
                j = i * 5
                exo.plot_exo(lh[j],lk[j],rh[j],rk[j],terrain_type)
                plt.savefig(filename)
                plt.close
                image = imageio.imread(filename)
                writer.append_data(image)
                    
            if with_ankle_constrain is True:
                lh = track2[1,:]
                lk = track2[3,:]
                rh = track2[0,:]
                rk = track2[2,:]
            else:
                lh = track4[1,:]
                lk = track4[3,:]
                rh = track4[0,:]
                rk = track4[2,:]
                
            exo.update_stace_leg(False)
            for i in range(num//5):
                j = i * 5
                exo.plot_exo(lh[j],lk[j],rh[j],rk[j],terrain_type)
                plt.savefig(filename)
                plt.close
                image = imageio.imread(filename)
                writer.append_data(image)
        os.remove(filename)       
        plt.show()   
    
elif terrain_type == "levelground":
    y = y_des[:,0] + [0,0,0,0]
    new_scale = np.ones(y.shape[0]) + [0,0,0,0]
    goal_offset = np.zeros(y.shape[0]) + [0,0,0,0]
    
    # For leve ground walking, it it to adjust the ending position of swing and stance hip joint for step length adjustment
    tmp_st_hip, tmp_sw_hip = CorrectHipForStepLength(y_des[:,-1],thigh_length,shank_length,step_length)
    goal_offset[0] = tmp_st_hip - y_des[0,-1]
    goal_offset[1] = tmp_sw_hip - y_des[1,-1]
    
    # Hip dmp unit needs the extral scale to correct the affect from the goal offset
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    # new_scale[3] = (dmp_gait.goal[1]+goal_offset[1])/(dmp_gait.goal[1])
    new_scale[3] = step_length / normal_step_len
    track,track_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)
    
    # second step
    y = np.array([track[1,-1],track[0,-1],track[3,-1],track[2,-1]])
    tmp_st_hip, tmp_sw_hip = CorrectHipForStepLength(y_des[:,-1],thigh_length,shank_length,step_length)
    goal_offset[0] = tmp_st_hip - y_des[0,-1]
    goal_offset[1] = tmp_sw_hip - y_des[1,-1]
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    new_scale[3] = step_length / normal_step_len
    track2,track2_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)

    # Plot original and updated trajectories
    st_hip=track[0,:]
    sw_hip=track[1,:]
    st_knee=track[2,:]
    sw_knee=track[3,:]
    
    f, axs = plt.subplots(2,2)
    axs[0][0].plot(y_des_time, raw_st_hip, label='raw st_hip')
    axs[0][0].plot(track_time, st_hip, label='dmp st_hip')
    axs[1][0].plot(y_des_time, raw_st_knee, label='raw st_knee')
    axs[1][0].plot(track_time, st_knee, label='dmp st_knee')

    axs[0][1].plot(y_des_time, raw_sw_hip, label='raw sw_hip')
    axs[0][1].plot(track_time, sw_hip, label='dmp sw_hip')
    axs[1][1].plot(y_des_time, raw_sw_knee, label='raw sw_knee')
    axs[1][1].plot(track_time, sw_knee, label='dmp sw_knee')
    for ax in axs:
        for a in ax:
            a.legend()
    
    st_hip=track2[0,:]
    sw_hip=track2[1,:]
    st_knee=track2[2,:]
    sw_knee=track2[3,:]
    f2, axs2 = plt.subplots(2,2)
    axs2[0][0].plot(y_des_time, raw_st_hip, label='raw st_hip')
    axs2[0][0].plot(track_time, st_hip, label='dmp st_hip')
    axs2[1][0].plot(y_des_time, raw_st_knee, label='raw st_knee')
    axs2[1][0].plot(track_time, st_knee, label='dmp st_knee')

    axs2[0][1].plot(y_des_time, raw_sw_hip, label='raw sw_hip')
    axs2[0][1].plot(track_time, sw_hip, label='dmp sw_hip')
    axs2[1][1].plot(y_des_time, raw_sw_knee, label='raw sw_knee')
    axs2[1][1].plot(track_time, sw_knee, label='dmp sw_knee')
    for ax in axs2:
        for a in ax:
            a.legend()
    
    # plot one step
    plt.figure(3)
    lh = track[0,:]
    lk = track[2,:]
    rh = track[1,:]
    rk = track[3,:]
    exo.plot_one_step(lh,lk,rh,rk,terrain_type)
    
    lh = track2[1,:]
    lk = track2[3,:]
    rh = track2[0,:]
    rk = track2[2,:]
    exo.update_stace_leg(False)
    exo.plot_one_step(lh,lk,rh,rk,terrain_type)
    plt.show()
    
    # # save to gif
    # plt.figure(3)
    # filename = 'fig.png'
    # with imageio.get_writer('levelground_walking.gif', mode='I') as writer:
    #     lh = track[0,:]
    #     lk = track[2,:]
    #     rh = track[1,:]
    #     rk = track[3,:]
    #     for i in range(num//5):
    #         j = i * 5
    #         exo.plot_exo(lh[j],lk[j],rh[j],rk[j],terrain_type)
    #         plt.savefig(filename)
    #         plt.close
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    #     lh = track2[1,:]
    #     lk = track2[3,:]
    #     rh = track2[0,:]
    #     rk = track2[2,:]
    #     exo.update_stace_leg(False)
    #     for i in range(num//5):
    #         j = i * 5
    #         exo.plot_exo(lh[j],lk[j],rh[j],rk[j],terrain_type)
    #         plt.savefig(filename)
    #         plt.close
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    # os.remove(filename)       
    # plt.show()
else:
    exo1 = EXO_SIM(obstacle_dist,obstacle_height,obstacle_width,thigh_length,shank_length,slope=slope)
    exo1.update_stace_leg(True)
    exo2 = EXO_SIM(obstacle_dist,obstacle_height,obstacle_width,thigh_length,shank_length,slope=-slope)
    exo2.update_stace_leg(True)
    
    # plot one step
    plt.figure(1)
    lh = raw_st_hip
    lk = raw_st_knee
    rh = raw_sw_hip
    rk = raw_sw_knee
    exo.plot_one_step(lh,lk,rh,rk,'levelground')
    
    lh = raw_sw_hip
    lk = raw_sw_knee
    rh = raw_st_hip
    rk = raw_st_knee
    exo.update_stace_leg(False)
    exo.plot_one_step(lh,lk,rh,rk,'levelground')
    
    plt.figure(2)
    lh = upslope_st_hip
    lk = upslope_st_knee
    rh = upslope_sw_hip
    rk = upslope_sw_knee
    exo1.plot_one_step(lh,lk,rh,rk,'slope')
    
    lh = upslope_sw_hip
    lk = upslope_sw_knee
    rh = upslope_st_hip
    rk = upslope_st_knee
    exo1.update_stace_leg(False)
    exo1.plot_one_step(lh,lk,rh,rk,'slope')
    
    plt.figure(3)
    lh = downslope_st_hip
    lk = downslope_st_knee
    rh = downslope_sw_hip
    rk = downslope_sw_knee
    exo2.plot_one_step(lh,lk,rh,rk,'slope')
    
    lh = downslope_sw_hip
    lk = downslope_sw_knee
    rh = downslope_st_hip
    rk = downslope_st_knee
    exo2.update_stace_leg(False)
    exo2.plot_one_step(lh,lk,rh,rk,'slope')
    plt.show()
    
    # # save to gif
    # plt.figure(3)
    # filename = 'fig.png'
    # with imageio.get_writer('5slope_raw_data.gif', mode='I') as writer:
    #     lh = slope_st_hip
    #     lk = slope_st_knee
    #     rh = slope_sw_hip
    #     rk = slope_sw_knee
    #     for i in range(num//5):
    #         j = i * 5
    #         exo.plot_exo(lh[j],lk[j],rh[j],rk[j],'slope')
    #         plt.savefig(filename)
    #         plt.close
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    #     lh = slope_sw_hip
    #     lk = slope_sw_knee
    #     rh = slope_st_hip
    #     rk = slope_st_knee
    #     exo.update_stace_leg(False)
    #     for i in range(num//5):
    #         j = i * 5
    #         exo.plot_exo(lh[j],lk[j],rh[j],rk[j],'slope')
    #         plt.savefig(filename)
    #         plt.close
    #         image = imageio.imread(filename)
    #         writer.append_data(image)
    # os.remove(filename)       
    # plt.show()
