from re import template
from sim import EXO_SIM
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import imageio
import os
import sys

from dmp import DMP
from utils import ReadDataFromText, Kinematics, InverseKinematics, JacobianMatrix

is_saving_gif = False
thigh_length = 0.44 #0.44
shank_length = 0.37 #0.565
obs_num = 1   # the obstacle number whose trajectory is selected as template 
template_type = 'mode3' 
"""
 mode0: select st_knee angle and hip px as templates, and can adjust the ending value of hip_px to satisfy inverse kinematics solution
 mode1: select st_hip angle and hip px as templates, and adjust the ending value of hip_px to satisfy inverse kinematics solution
 mode2: px + pz, set the ending value of px as half of step length, and adjust the ending value of pz
 mode3: select st_ankle angle and hip px as templates
"""
# mode3 modulated parameters
delta_h = 0.056 # the declination in vertical direction in the ending of swing phase


subj_tight_shank = np.array([[0.410,0.374],[0.507,0.415],[0.518,0.424],[0.497,0.459],[0.467,0.459]])
obst_size = np.array([[0.05,0.08],[0.125,0.145],[0.125,0.20],[0.105,0.274],[0.20,0.125],[0.125,0.145]])


def PxKnee2AngleCal(track, tight, shank):
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

def PxPz2AngleCal(track,tight,shank):
    st_hip_px = track[0,:]
    st_hip_pz = tight + shank + track[1,:]
    sw_foot_px=track[2,:]
    sw_foot_pz=track[3,:]
    
    st_hip, st_knee = InverseKinematics(np.array([-st_hip_px,-st_hip_pz]),tight,shank)
    sw_hip, sw_knee = InverseKinematics(np.array([-st_hip_px+sw_foot_px,-st_hip_pz+sw_foot_pz]),tight,shank)
    return st_hip, st_knee, sw_hip, sw_knee

def PxHip2AngleCal(track,tight,shank):
    st_hip = track[0,:]
    st_hip_px = track[1,:]
    sw_foot_px = track[2,:] # the position of swing foot relative to the stance foot
    sw_foot_pz = track[3,:]
    
    delta_px = -st_hip_px - tight * np.sin(st_hip * np.pi/180)
    global_knee_angle = np.arcsin(delta_px/shank)
    st_hip_pz = tight * np.cos(st_hip * np.pi/180) + shank * np.cos(global_knee_angle)
    
    st_knee = st_hip - global_knee_angle * 180 / np.pi
    sw_hip, sw_knee = InverseKinematics(np.array([-st_hip_px+sw_foot_px,-st_hip_pz+sw_foot_pz]),tight,shank)
    return st_hip, st_knee, sw_hip, sw_knee

def PxAnkle2AngleCal(track,tight,shank):
    st_ankle = track[0,:]
    st_hip_px = track[1,:]
    sw_foot_px = track[2,:] # the position of swing foot relative to the stance foot
    sw_foot_pz = track[3,:]
    
    delta_px = st_hip_px - shank * np.sin(- st_ankle * np.pi/180)
    st_hip = -np.arcsin(delta_px/tight) * 180 / np.pi
    st_knee = st_hip - st_ankle
    st_hip_pz = shank * np.cos(st_ankle*np.pi/180) + tight * np.cos(st_hip * np.pi/180)
    
    sw_hip, sw_knee = InverseKinematics(np.array([-st_hip_px+sw_foot_px,-st_hip_pz+sw_foot_pz]),tight,shank)
    return st_hip, st_knee, sw_hip, sw_knee

def Obst2Dmp(obs_h,obs_w,y0_sw,peak_sw,dmp_sw,y0_st,dmp_st):
    """using obstacle size to calculate the parameters provided to the template DMP unit 

    Args:
        obs_h (float): obstacle height
        obs_w (float): obstacle width
        y0_sw (1D-array): the initial values for swing foot position (Px0, Pz0)
        peak_sw (float): the peak value of swing foot position Pz
        dmp_sw (dmp object): the dmp template for swing leg (px, pz)
        y0_st (1D-array): the initial values for stance leg, stance knee angle, hip position Px
        dmp_st (dmp object): the dmp template for stance leg (knee angle, hip px)
    """
    Delta_L1 = 0.2
    Delta_L2 = 0.3
    Delta_H = 0.2
    step_len = obs_w + Delta_L1 + Delta_L2
    step_height = obs_h + Delta_H
    

    # swing dmp: convert step length and height to DMP parameters
    new_scale = np.ones(y0_sw.shape[0])
    goal_offset = np.zeros(y0_sw.shape[0])
    
    goal_sw_dmp = dmp_sw.goal
    y0_sw_dmp = dmp_sw.y0
        # Px --- step length, adjusting goal_offset, but the adjustement of scaling is to avoid to distort the shape 
    goal_offset[0] = step_len - goal_sw_dmp[0]
    new_scale[0] = (goal_sw_dmp[0] + goal_offset[0] - y0_sw[0])/(goal_sw_dmp[0] - y0_sw_dmp[0])
        # Pz --- step height
    goal_offset[1] = 0
    new_scale[1] = (step_height - goal_sw_dmp[1] - goal_offset[1]) / (peak_sw - goal_sw_dmp[1])
    track_sw, _, =  dmp_sw_temp.full_generation(num_steps=500,y0=y0_sw,new_scale=new_scale,goal_offset=goal_offset)
    
    # stance dmp: 
    new_scale = np.ones(y0_st.shape[0])
    goal_offset = np.zeros(y0_st.shape[0])
    goal_st_dmp = dmp_st.goal
    y0_st_dmp = dmp_st.y0
    
    # changing goal_offset, which is related to 
    goal_offset[1] =  step_len / 2 - goal_st_dmp[1]
    new_scale[1] = (goal_st_dmp[1] + goal_offset[1] - y0_st[1])/(goal_st_dmp[1] - y0_st_dmp[1])
    track_st, _, = dmp_st_temp.full_generation(num_steps=500,y0=y0_st,new_scale=new_scale,goal_offset=goal_offset)
    
    return track_sw,track_st
    

# load mocap data
with open('obst_first_step_all_subj_data_split.npy','rb') as f:
    all_subj_st_hip = np.load(f)
    all_subj_sw_hip = np.load(f)
    all_subj_st_knee = np.load(f)
    all_subj_sw_knee = np.load(f)
    all_subj_st_px = np.load(f) # stance foot position relative to hip joint
    all_subj_st_pz = np.load(f)
    all_subj_sw_px = np.load(f)
    all_subj_sw_pz = np.load(f)
    
all_subj_relative_ankle_px = all_subj_sw_px - all_subj_st_px
all_subj_relative_ankle_pz = all_subj_sw_pz - all_subj_st_pz
all_subj_hip_px = np.zeros(all_subj_relative_ankle_px.shape) # hip position relative to the hightest point of lower limb in standing straight
all_subj_hip_pz = np.zeros(all_subj_relative_ankle_px.shape)
for i in range(all_subj_sw_px.shape[0]):
    for j in range(5):
        all_subj_hip_px[i,j,:] = -all_subj_st_px[i,j,:]   # hip position relative to the stance foot
        all_subj_hip_pz[i,j,:] = -all_subj_st_pz[i,j,:] + all_subj_st_pz[i,j,0]

mean_subj_st_hip = all_subj_st_hip.mean(axis=1)
mean_subj_sw_hip = all_subj_sw_hip.mean(axis=1)
mean_subj_st_knee = all_subj_st_knee.mean(axis=1)
mean_subj_sw_knee = all_subj_sw_knee.mean(axis=1)
mean_subj_relative_px = all_subj_relative_ankle_px.mean(axis=1)
mean_subj_relative_pz = all_subj_relative_ankle_pz.mean(axis=1)
mean_subj_hip_px = all_subj_hip_px.mean(axis=1)
mean_subj_hip_pz = all_subj_hip_pz.mean(axis=1)


""" optional templates and fitting it with DMP """
# trajectory templates
raw_st_hip = np.array(mean_subj_st_hip.mean(axis=0))
raw_st_knee = np.array(mean_subj_st_knee.mean(axis=0)) # stance knee
raw_st_ankle = raw_st_hip - raw_st_knee
raw_st_hip_px = np.array(mean_subj_hip_px[obs_num,:]) # hip position relative to the hightest point of lower limb in standing straight
raw_st_hip_pz = np.array(mean_subj_hip_pz[obs_num,:])
raw_sw_px = np.array(mean_subj_relative_px[obs_num,:])
raw_sw_pz = np.array(mean_subj_relative_pz[obs_num,:])
peak_sw_pz =  np.amax(raw_sw_pz)

# Plot templates
f0,axs0 = plt.subplots(2,3)
f = f0
axs = axs0
f.suptitle('seleted templates')
axs[0][0].plot(raw_st_hip[0:500:10],'*',label='st_hip angle')
axs[0][0].plot(raw_st_knee[0:500:10],'*',label='st_knee angle')
axs[0][0].plot(raw_st_ankle[0:500:10],'*',label='st_ankle angle')
axs[1][0].plot(raw_st_hip[0:500:10],raw_st_knee[0:500:10],'*',label='hip-knee agnle')

axs[0][1].plot(raw_st_hip_px[0:500:10],'*',label='st_hip_px')
axs[0][1].plot(raw_st_hip_pz[0:500:10],'*',label='st_hip_pz')
axs[1][1].plot(raw_st_hip_px[0:500:10],raw_st_hip_pz[0:500:10],'*',label='st hip position trajectory')

axs[0][2].plot(raw_sw_px[0:500:10],'*',label='sw_ankle_px')
axs[0][2].plot(raw_sw_pz[0:500:10],'*',label='sw_ankle_pz')
axs[1][2].plot(raw_sw_px[0:500:10],raw_sw_pz[0:500:10],'*',label='swing ankle position trajectory')
for ax in axs:
    for a in ax:
        a.legend()


""" generate gait trajectory for different subjects with different obstacles"""
# create and initial exo simulation models
list_obstacle = [0.05, 0.125, 0.2]
is_shank_first = True
terrain_type = "obstacle"
slope = 10.0
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
    
    """ DMP learning with templates """
    # templates for swing leg
    y_sw_des = np.array([raw_sw_px, raw_sw_pz])  # for a specific obstacle, the template for swing leg is also specific
    y_time = np.arange(0, raw_sw_pz.shape[0])*0.001
    ay = np.ones(4) * 50
    by = ay / 4
    dmp_sw_temp = DMP(y_sw_des, y_time, n_bfs=200, dt=0.001, isz=True)
    _, _, = dmp_sw_temp.imitate_path()
    
    # templates for stance leg
    if template_type is 'mode0':
        y_st_des = np.array([raw_st_knee, raw_st_hip_px])
    elif template_type is 'mode1':
        y_st_des = np.array([raw_st_hip, raw_st_hip_px])
    elif template_type is 'mode2':
        y_st_des = np.array([raw_st_hip_px, raw_st_hip_pz])
    elif template_type is 'mode3':
        y_st_des = np.array([raw_st_ankle, raw_st_hip_px])

        
    dmp_st_temp = DMP(y_st_des, y_time, n_bfs=200, dt=0.001, isz=True)
    _, _, = dmp_st_temp.imitate_path()


    step_len = 0.6
    step_height = 0.3
    """ DMP generattion """
    # swing leg
    goal = dmp_sw_temp.goal
    y0_sw = y_sw_des[:,0] + [0,0]
    new_scale = np.ones(y0_sw.shape[0]) + [0.0,0.0]
    goal_offset = np.zeros(y0_sw.shape[0]) + [0.0,0.0]
    
    goal_offset[0] = step_len - dmp_sw_temp.goal[0]
    new_scale[0] = (dmp_sw_temp.goal[0] + goal_offset[0] - y0_sw[0])/(dmp_sw_temp.goal[0] - dmp_sw_temp.y0[0])
    new_scale[1] = (step_height - dmp_sw_temp.goal[1] - goal_offset[1]) / (peak_sw_pz - dmp_sw_temp.goal[1])
    
    track_sw, time_sw, =  dmp_sw_temp.full_generation(num_steps=500,y0=y0_sw,new_scale=new_scale,goal_offset=goal_offset)
    sw_foot_px = track_sw[0,:]
    sw_foot_pz = track_sw[1,:]
    
    """
    # test the relationship between peak value and scaling, goal
    # change the scaling or offset by a fixed increasement, and record the corresponding peak values
    # the results show that peak value actually has linear relationship with the scaling (just for parabolic curves), which is accordance with the following equation
    peak = np.zeros(21)
    d_peak = np.zeros(21)
    goal = np.copy(dmp_sw_temp.goal)
    init_peak = sw_foot_pz[np.argmax(sw_foot_pz)]
    for i in range(21):
        # for monotonous curves including sw_hip and sw_px trajectories, the goal offset will affect the scaling
        # y0_sw[0] = y_sw_des[0,0] + i * 0.005
        goal_offset[0] -= i * 0.001
        new_scale[0] = (goal[0] + goal_offset[0] - y0_sw[0])/(goal[0] - y_sw_des[0,0])
        
        # for parabolic curves like sw_knee and sw_pz trajectories, the peak value is changed with the scaling
        new_scale[1] = 0.0 + i * 0.1
        y0_sw[1] += i * 0.00
        goal_offset[1] = i * 0.002
        # new_scale[1] = (init_peak - goal[1] - goal_offset[1]) / (init_peak - goal[1])
        
        # calculated peak value, the equation describe the relationship between peak value and DMP parameters
        d_peak[i] = new_scale[1] * (init_peak - goal[1]) + goal[1] + goal_offset[1]
        
        track_sw, time_sw, =  dmp_sw_temp.full_generation(num_steps=500,y0=y0_sw,new_scale=new_scale,goal_offset=goal_offset)
        sw_foot_px = track_sw[0,:]
        sw_foot_pz = track_sw[1,:]
        
        index = np.argmax(sw_foot_pz)
        peak[i] = sw_foot_pz[index]
        print(new_scale[1],index,sw_foot_pz[index],new_scale[1])
        plt.figure(2)
        plt.plot(sw_foot_pz)
        plt.figure(3)
        plt.plot(sw_foot_px)
    
    plt.figure(4)
    plt.plot(peak)
    plt.plot(d_peak)
    
    plt.show()
    sys.exit("Error message")
    """
    
    if template_type is 'mode0':
        y0_st = y_st_des[:,0] + [0,0]
        new_scale = np.ones(y0_sw.shape[0]) + [0.0,0.0]
        goal_offset = np.zeros(y0_sw.shape[0]) + [0.0,0.0]    
        # goal_offset[1] += 0.00
        goal_offset[1] =  step_len / 2 - dmp_st_temp.goal[1]
        new_scale[1] = (goal_offset[1] + dmp_st_temp.goal[1] - y0_st[1])/(dmp_st_temp.goal[1]-dmp_st_temp.y0[1])
        track_st, st_time = dmp_st_temp.full_generation(num_steps=500,y0=y0_st,new_scale=new_scale,goal_offset=goal_offset)
        
        st_hip,st_knee,sw_hip,sw_knee = PxKnee2AngleCal(np.vstack((track_st, track_sw)), thigh_length, shank_length)

    elif template_type is 'mode1':
        y0_st = y_st_des[:,0] + [0,0]
        new_scale = np.ones(y0_sw.shape[0]) + [0.0,0.0]
        goal_offset = np.zeros(y0_sw.shape[0]) + [0.0,0.0]
        track_st, st_time = dmp_st_temp.full_generation(num_steps=500,y0=y0_st,new_scale=new_scale,goal_offset=goal_offset)
        
        st_hip,st_knee,sw_hip,sw_knee = PxHip2AngleCal(np.vstack((track_st, track_sw)), thigh_length, shank_length)

    elif template_type is 'mode2':
        y0_st = y_st_des[:,0] + [0,-0.005]
        new_scale = np.ones(y0_sw.shape[0]) + [0.0,0.0]
        goal_offset = np.zeros(y0_sw.shape[0]) + [0.0,-0.01]
        track_st, st_time = dmp_st_temp.full_generation(num_steps=500,y0=y0_st,new_scale=new_scale,goal_offset=goal_offset)
        
        st_hip,st_knee,sw_hip,sw_knee = PxPz2AngleCal(np.vstack((track_st, track_sw)),thigh_length, shank_length)
       
    elif template_type is 'mode3':
        # cal the ending status of stance leg
        # one way is to specify the ending knee angle
        knee_ending_angle = raw_st_knee[0] / 180 * np.pi
        hip_pz = np.sqrt((thigh_length**2 + shank_length**2+ 2*np.cos(knee_ending_angle)*thigh_length*shank_length-step_len*step_len/4))
        tmp_hip, tmp_knee = InverseKinematics(np.array([-step_len / 2,-hip_pz]),thigh_length,shank_length)
        
        # the other way is to specify the vertical distance
        # tmp_hip, tmp_knee = InverseKinematics(np.array([-step_len / 2,delta_h-thigh_length-shank_length]),thigh_length,shank_length)
       
        goal_ankle = tmp_hip - tmp_knee
        
        y0_st = y_st_des[:,0] + [0,0]
        new_scale = np.ones(y0_sw.shape[0]) + [0.0,0.0]
        goal_offset = np.zeros(y0_sw.shape[0]) + [0.0,0.0]
        
        goal_offset[1] =  step_len / 2 - dmp_st_temp.goal[1]
        new_scale[1] = (goal_offset[1] + dmp_st_temp.goal[1] - y0_st[1])/(dmp_st_temp.goal[1]-dmp_st_temp.y0[1])
        goal_offset[0] =  goal_ankle - dmp_st_temp.goal[0]
        new_scale[0] = (goal_offset[0] + dmp_st_temp.goal[0] - y0_st[0])/(dmp_st_temp.goal[0]-dmp_st_temp.y0[0])
        
        track_st, st_time = dmp_st_temp.full_generation(num_steps=500,y0=y0_st,new_scale=new_scale,goal_offset=goal_offset)
        
        st_hip,st_knee,sw_hip,sw_knee = PxAnkle2AngleCal(np.vstack((track_st, track_sw)),thigh_length, shank_length)
        
    st_hip_px,st_hip_pz = Kinematics(st_hip,st_knee,thigh_length, shank_length)
    st_hip_px = -st_hip_px
    st_hip_pz = -st_hip_pz - thigh_length - shank_length
    
    print(np.amax(-st_hip),np.amax(st_knee),np.amax(sw_hip),np.amax(sw_knee))
        
    f1, axs1=plt.subplots(2,3)
    axs = axs1
    f = f1
    axs[0][0].set_ylabel('Position/[m]')
    axs[0][0].plot(raw_st_hip_px,'r--',label='template hip px')
    axs[0][0].plot(st_hip_px,'r',label='generated hip px')
    # axs[0][0].plot(mean_subj_hip_px[obs_num,:],'c',label='gound truth hip px')
    axs[0][0].plot(raw_st_hip_pz,'b--',label='template hip pz')
    axs[0][0].plot(st_hip_pz,'b',label='generated hip pz')
    # axs[1][0].plot(mean_subj_hip_pz[obs_num,:],'c',label='gound truth hip pz')
    axs[1][0].set_xlabel('Px/[m]')
    axs[1][0].set_ylabel('Pz/[m]')
    axs[1][0].plot(raw_st_hip_px,raw_st_hip_pz,'c--',label='template hip position')
    axs[1][0].plot(st_hip_px,st_hip_pz,'c',label='generated hip position')
    
    
    axs[0][1].set_ylabel('Position/[m]]')
    axs[0][1].plot(raw_sw_px,'r--',label='template sw_px')
    axs[0][1].plot(sw_foot_px,'r',label='dmp output sw_px')
    # axs[0][1].plot(mean_subj_relative_px[obs_num,:],'c',label='ground truth sw_px')
    axs[0][1].plot(raw_sw_pz,'b--',label='template sw_pz')
    axs[0][1].plot(sw_foot_pz,'b',label='dmp output sw_pz')
    # axs[1][1].plot(mean_subj_relative_pz[obs_num,:],'c',label='ground truth sw_pz')
    axs[1][1].set_xlabel('Px/[m]')
    axs[1][1].set_ylabel('Pz/[m]]')
    axs[1][1].plot(raw_sw_px,raw_sw_pz,'c--',label='template sw ankle')
    axs[1][1].plot(sw_foot_px,sw_foot_pz,'c',label='generated sw ankle')
    
    
    axs[0][2].set_ylabel('stance angle/[deg]')
    axs[0][2].plot(raw_st_hip[0:500:10],'r--',label='template st_hip')
    axs[0][2].plot(raw_st_knee[0:500:10],'b--',label='template st_knee')
    axs[0][2].plot(st_hip[0:500:10],'r*',label='generated st_hip')
    axs[0][2].plot(st_knee[0:500:10],'b*',label='generated st_knee')
    axs[1][2].set_ylabel('swing angle/[deg]')
    axs[1][2].plot(sw_hip[0:500:10],'r*',label='sw_hip')
    axs[1][2].plot(sw_knee[0:500:10],'b*',label='sw_knee')
    for ax in axs:
        for a in ax:
            a.legend()
            
    # # plot the generated swing foot trajectory with different obstacles
    # index = [0,1,4,1,2,3]
    # f2, axs2=plt.subplots(2,3)
    # f = f2
    # axs = axs2
    # axs[0][0].set_title('5x10')
    # axs[0][1].set_title('10x10')
    # axs[0][2].set_title('20x10')
    # axs[1][0].set_title('10x10')
    # axs[1][1].set_title('10x20')
    # axs[1][2].set_title('10x30')
    
    # peak_sw_pz =  mean_subj_relative_pz[obs_num,np.argmax(mean_subj_relative_pz[obs_num,:])]
    
    # for i in range(6):
    #     row = i // 3
    #     col = i - row * 3
    #     k = index[i]
    #     axs[row][col].set_xlim([-0.05, 0.75])
    #     axs[row][col].set_ylim([-0.05, 0.60])
        
        
    #     track_sw, track_st = Obst2Dmp(obst_size[k][0],obst_size[k][1],y0_sw,peak_sw_pz,dmp_sw_temp,y0_st,dmp_st_temp)
    #     sw_foot_px = track_sw[0,:]
    #     sw_foot_pz = track_sw[1,:]
        
    #     axs[row][col].plot(mean_subj_relative_px[k,:],mean_subj_relative_pz[k,:],'r--',label='gound truth')
    #     axs[row][col].plot(sw_foot_px,sw_foot_pz,'b',label='generated')
        
    #     obstacle_height = obst_size[k][0]
    #     obstacle_width = obst_size[k][1]
    #     obstacle_px = np.array([obstacle_dist,obstacle_dist,obstacle_dist+obstacle_width,obstacle_dist+obstacle_width,obstacle_dist])
    #     obstacle_pz = np.array([0,obstacle_height,obstacle_height,0,0])
    #     axs[row][col].plot(obstacle_px,obstacle_pz,'r')
    # for ax in axs:
    #     for a in ax:
    #         a.legend()
    #         a.grid(True)
    #         a.set_xlabel('PX/[m]]')
    #         a.set_ylabel('PZ/[m]]')
    

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