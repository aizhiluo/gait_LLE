from sim import EXO_SIM
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import imageio
import os

from dmp import DMP
from utils import ReadDataFromText, Kinematics, InverseKinematics, JacobianMatrix


def KneePx2AngleCal(track, tight, shank):
    """Selecting hip_px st_knee, swing ankle px and pz as the templates, 
        and calculate the joint angle using kinematics and inverse kinematics

    Args:
        track (4d-array): templates
        tight (float): tight length
        shank (float): shank length

    Returns:
        float: four joint angles
    """
    st_knee=track[0,:]
    st_hip_px=track[1,:]
    sw_foot_px=track[2,:] # the position of swing foot relative to the stance foot
    sw_foot_pz=track[3,:]
    
    # calculate stance hip angle and PZ according PX and stance knee angle
    N = len(st_knee)
    st_hip = np.zeros(st_knee.shape)
    st_hip_pz = np.zeros(st_knee.shape)
    for i in range(N):
        dist = np.sqrt(tight*tight + shank*shank - 2*np.cos(np.pi-st_knee[i]*np.pi/180)*tight*shank)
        ang_s = np.arcsin(shank * np.sin(st_knee[i]*np.pi/180) / dist)
        ang_alpha = -np.arcsin(st_hip_px[i]/dist)
        st_hip[i] = (ang_alpha + ang_s) / np.pi * 180
        st_hip_pz[i] = dist * np.cos(ang_alpha)
    
    temp_px = sw_foot_px - st_hip_px
    temp_pz = sw_foot_pz - st_hip_pz
    sw_hip, sw_knee = InverseKinematics(np.array([temp_px,temp_pz]),tight,shank)
    
    return st_hip, st_knee, sw_hip, sw_knee
    
is_saving_gif = False
subj_tight_shank = [[0.410,0.374],[0.507,0.415],[0.518,0.424],[0.497,0.459],[0.467,0.459]]

# load mocap data
with open('obst_first_step_all_subj_data.npy','rb') as f:
    all_subj_st_hip = np.load(f)
    all_subj_sw_hip = np.load(f)
    all_subj_st_knee = np.load(f)
    all_subj_sw_knee = np.load(f)
    all_subj_st_px = np.load(f)
    all_subj_st_pz = np.load(f)
    all_subj_sw_px = np.load(f)
    all_subj_sw_pz = np.load(f)
    
all_subj_relative_ankle_px = all_subj_sw_px - all_subj_st_px
all_subj_relative_ankle_pz = all_subj_sw_pz - all_subj_st_pz
all_subj_hip_px = np.zeros(all_subj_relative_ankle_px.shape)
all_subj_hip_pz = np.zeros(all_subj_relative_ankle_px.shape)
for i in range(5):
    for j in range(5):
        all_subj_hip_px[i,j,:] = -all_subj_st_px[i,j,:]
        all_subj_hip_pz[i,j,:] = -all_subj_st_pz[i,j,:]

mean_subj_st_hip = all_subj_st_hip.mean(axis=1)
mean_subj_sw_hip = all_subj_sw_hip.mean(axis=1)
mean_subj_st_knee = all_subj_st_knee.mean(axis=1)
mean_subj_sw_knee = all_subj_sw_knee.mean(axis=1)
mean_subj_relative_px = all_subj_relative_ankle_px.mean(axis=1)
mean_subj_relative_pz = all_subj_relative_ankle_pz.mean(axis=1)
mean_subj_hip_px = all_subj_hip_px.mean(axis=1)
mean_subj_hip_pz = all_subj_hip_pz.mean(axis=1)


""" select template and fitting it with DMP """
obs_num =0
# DMP creating and training
raw_st_knee = np.array(mean_subj_st_knee.mean(axis=0)) # stance knee
raw_st_hip_px = np.array(mean_subj_hip_px[obs_num,:])
raw_sw_px = np.array(mean_subj_relative_px[obs_num,:])
raw_sw_pz = np.array(mean_subj_relative_pz[obs_num,:])

y_des = np.array([raw_st_knee, raw_st_hip_px, raw_sw_px, raw_sw_pz])
y_des_time = np.arange(0, raw_sw_pz.shape[0])*0.001

ay = np.ones(4) * 50
by = ay / 4
dmp_gait = DMP(y_des, y_des_time, n_bfs=200, dt=0.001, isz=True)
_, _, = dmp_gait.imitate_path()

f1,axs1 = plt.subplots(2,2)
axs1[0][0].plot(raw_st_knee)
axs1[1][0].plot(raw_st_hip_px)
axs1[0][1].plot(raw_sw_px)
axs1[1][1].plot(raw_sw_pz)


""" generate gait trajectory for different subjects with different obstacles"""
# create and initial exo simulation models
list_obstacle = [0.05, 0.125, 0.2]
is_shank_first = True
terrain_type = "obstacle"
slope = 10.0
thigh_length = 0.51 #0.44
shank_length = 0.43 #0.565
obstacle_dist = 0.20 # assume obstacle locates in x-axis z = 0
obstacle_width = 0.15
obstacle_height = 0.2
obstacle_height = list_obstacle[0]
obstacle_px = np.array([obstacle_dist,obstacle_dist,obstacle_dist+obstacle_width,obstacle_dist+obstacle_width,obstacle_dist])
obstacle_pz = np.array([0,obstacle_height,obstacle_height,0,0])
exo = EXO_SIM(obstacle_dist,obstacle_height,obstacle_width,thigh_length,shank_length,slope=slope)
exo.update_stace_leg(True)


if terrain_type == "obstacle":
    
    obstacle_height = list_obstacle[1]
    obstacle_px = np.array([obstacle_dist,obstacle_dist,obstacle_dist+obstacle_width,obstacle_dist+obstacle_width,obstacle_dist])
    obstacle_pz = np.array([0,obstacle_height,obstacle_height,0,0])
    
    """ DMP generattion """
    y = y_des[:,0] + [0,0,0,0]
    new_scale = np.ones(y.shape[0]) + [0.0,0,0,0.0]
    goal_offset = np.zeros(y.shape[0]) + [0,0,0,0.0]
    track, track_time = dmp_gait.full_generation(num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)
    st_knee=track[0,:]
    st_hip_px=track[1,:]
    sw_foot_px=track[2,:] # the position of swing foot relative to the stance foot
    sw_foot_pz=track[3,:]
    
    # cal joint angles based on (st_knee angle, st_hip_px, sw_ankle_px, sw_ankle_pz)
    st_hip,st_knee,sw_hip,sw_knee = KneePx2AngleCal(track, thigh_length, shank_length)

    obs_num = 0
    
    f, axs=plt.subplots(2,3)
    axs[0][0].plot(raw_st_knee,'r',label='template st knee')
    axs[0][0].plot(st_knee,'b',label='dmp stance knee')
    axs[0][0].plot(mean_subj_st_knee[obs_num,:],'c',label='gound truth st_knee')
    axs[0][0].set_ylabel('angle/[deg]')
    axs[1][0].plot(raw_st_hip_px,'r',label='template st_pz')
    axs[1][0].plot(st_hip_px,'b',label='dmp stance px')
    axs[1][0].plot(mean_subj_hip_px[obs_num,:],'c',label='gound truth st_pz')
    axs[1][0].set_ylabel('Px/[m]')
    
    axs[0][1].plot(raw_sw_px,'r',label='template sw_px')
    axs[0][1].plot(sw_foot_px,'b',label='dmp output sw_px')
    axs[0][1].plot(mean_subj_relative_px[obs_num,:],'c',label='ground truth sw_px')
    axs[0][1].set_ylabel('PX/[m]]')
    axs[1][1].plot(raw_sw_pz,'r',label='template sw_pz')
    axs[1][1].plot(sw_foot_pz,'b',label='dmp output sw_pz')
    axs[1][1].plot(mean_subj_relative_pz[obs_num,:],'c',label='ground truth sw_pz')
    axs[1][1].set_ylabel('PZ/[m]]')
    
    axs[0][2].plot(st_hip,'r',label='st_hip')
    axs[0][2].plot(st_knee,'b',label='st_knee')
    axs[0][2].set_ylabel('stance angle/[deg]')
    axs[1][2].plot(sw_hip,'r',label='sw_hip')
    axs[1][2].plot(sw_knee,'b',label='sw_knee')
    axs[1][2].set_ylabel('swing angle/[deg]')
    
    for ax in axs:
        for a in ax:
            a.legend()
            
    
    index = [0,1,4,1,2,3]
    f2, axs2=plt.subplots(2,3)
    axs2[0][0].set_title('5x10')
    axs2[0][1].set_title('10x10')
    axs2[0][2].set_title('20x10')
    axs2[1][0].set_title('10x10')
    axs2[1][1].set_title('10x20')
    axs2[1][2].set_title('10x30')
    for i in range(6):
        row = i // 3
        col = i - row * 3
        k = index[i]
        # axs3[row][col].set_xlim([-0.05, 0.75])
        # axs3[row][col].set_ylim([-0.05, 0.60])
        axs2[row][col].plot(mean_subj_relative_px[k,:],mean_subj_relative_pz[k,:],'r--',label='gound truth')
        axs2[row][col].plot(sw_foot_px,sw_foot_pz,'b',label='generated')
        axs2[row][col].plot(obstacle_px,obstacle_pz,'r')
    for ax in axs2:
        for a in ax:
            a.legend()
            a.grid(True)
            a.set_xlabel('PX/[m]]')
            a.set_ylabel('PZ/[m]]')
    

    # save as gif file
    if is_saving_gif is True:  
        plt.figure(4)
        filename = 'fig.png'
        with imageio.get_writer('5x10_obstacle.gif', mode='I') as writer:
            lh = st_hip
            lk = st_knee
            rh = sw_hip
            rk = sw_knee
            for i in range(len(lh)//5):
                j = i * 5
                exo.plot_exo(lh[j],lk[j],rh[j],rk[j],terrain_type)
                plt.savefig(filename)
                plt.close
                image = imageio.imread(filename)
                writer.append_data(image)
        os.remove(filename)
        
    plt.show()