from sim import EXO_SIM
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import imageio
import os

from copy import deepcopy
from dmp import DMP
from utils import CorrectHipForStepLength, JointAngleForRamp, Kinematics, InverseKinematics, JacobianMatrix

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
        
        # #### online adjust swing leg trajectory based repulsive force with DMP ####
        # # distance between both legs
        # tmp_px,tmp_pz = Kinematics(y[0:2],y[2:4],thigh_length,shank_length)
        # px = tmp_px[1] - tmp_px[0]
        # pz = tmp_pz[1] - tmp_pz[0]
        
        # tmp_force_x,tmp_force_z = RepusiveForceFile(px,pz,0.15,0.1,0.1) # the obstacle info is fixed
        # sw_hip_angle = y[1]
        # sw_knee_angle = y[3]
        
        # # transfer force to joint torque by JacobianMatrix
        # JacoM = JacobianMatrix(sw_hip_angle,sw_knee_angle,thigh_length,shank_length)
        # tmp_torque_hip = JacoM[0,0] * tmp_force_x + JacoM[1,0] * tmp_force_z
        # tmp_torque_knee = JacoM[0,1] * tmp_force_x + JacoM[1,1] * tmp_force_z
        # force = np.array([0,tmp_torque_hip*40,0,tmp_torque_knee*20]) * 1.0
        # y += force
        
        track[:, i] = deepcopy(y)
              
    # dmp_st_hip = track[0,:]
    # dmp_sw_hip = track[1,:]
    # dmp_st_knee = track[2,:]
    # dmp_sw_knee = track[3,:]
    
    return track, track_time
      
def RepusiveForceFile(px,pz,obs_d,obs_w,obs_h):
    # check if input is array
    if type(px) == np.ndarray:
        force_x = np.zeros(px.shape[0])
        force_z = np.zeros(px.shape[0])
    else:
        force_x = 0.0
        force_z = 0.0

    # regard obstacle as one point
    only_one_point = True
    x_force_scale = 1.0
    
    # corner and center position of the obstacle
    corner_x = np.array([obs_d,obs_d,obs_d+obs_w,obs_d+obs_w])
    corner_z = np.array([0.0,obs_h,obs_h,0.0])
    center_x = np.mean(corner_x)
    center_z = np.mean(corner_z)
    
    # the parameters of the repulsive force
    radius = max(obs_h,obs_w) / np.sqrt(2)
    D = radius / 3.0 # tolent space
        
    alpha = 1.0
    k = 1 / radius
    beta = np.sqrt((radius-D)/(D*radius**3))
    # k = alpha / (radius + D) // the filed with this parameters is not monotone function
    # beta = np.sqrt(k/(radius * D))
    
    # potential  repulsive force fields formula
    if only_one_point is True:
        # only considering the center point of the obstacle
        dist = (px-center_x)**2 + (pz-center_z)**2
        force = alpha/(k+dist*beta*beta)
        force_x = force * (px-center_x)/np.sqrt(dist) * x_force_scale
        force_z = force * (pz-center_z)/np.sqrt(dist)
        
    else:
        # consider four corner of the obstacle
        for i in range(4):
            dist = (px-corner_x[i])**2 + (pz-corner_z[i])**2
            force = alpha/(k+dist*beta*beta)
            force_x += force * (px-corner_x[i])/np.sqrt(dist) * x_force_scale / 4
            force_z += force * (pz-corner_z[i])/np.sqrt(dist) / 4
    
    return force_x, (force_z + abs(force_z))/2.0


# load gait data
path = "D:/MyFolder/code/gait_ws/gait_LLE/data/joint_angle/"
file_name = "dmp_left_swing" #"SN_all_subj_obstacle_first" #
file_path = path + file_name + ".txt"
raw_st_hip, raw_sw_hip, raw_st_knee, raw_sw_knee=read_gait_data_txt(file_path)
num = raw_st_hip.shape[0]

# create and train DMP module
y_des = np.array([raw_st_hip, raw_sw_hip, raw_st_knee, raw_sw_knee])
y_des_time = np.arange(0, y_des.shape[1])*0.001
dmp_gait = DMP(y_des, y_des_time,n_bfs=250, dt=0.001, isz=True)
_, _, = dmp_gait.imitate_path()

# create and initial exo simulation models
terrain_type = "slope"
slope = 5.0
thigh_length = 0.44
shank_length = 0.565
obstacle_dist = 0.15 # assume obstacle locates in x-axis z = 0
obstacle_width = 0.1
obstacle_height = 0.1
obstacle_px = np.array([obstacle_dist,obstacle_dist,obstacle_dist+obstacle_width,obstacle_dist+obstacle_width,obstacle_dist])
obstacle_pz = np.array([0,obstacle_height,obstacle_height,0,0])
exo = EXO_SIM(obstacle_dist,obstacle_height,obstacle_width,thigh_length,shank_length,slope=slope)
exo.update_stace_leg(True)


# different terrain simulation
if terrain_type == "obstacle":
    # generate gait trajectory for obstacle first step
    y = y_des[:,0] + [0,0,0,0]
    new_scale = np.ones(y.shape[0]) + [0,0,0,0]
    goal_offset = np.zeros(y.shape[0]) + [0,0,0,0]
    track, track_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)
    
    # using repulsive force field to adjust swing leg trajectory
    st_hip=track[0,:]
    sw_hip=track[1,:]
    st_knee=track[2,:]
    sw_knee=track[3,:]
    _,_,updated_sw_h,updated_sw_k,force_x,force_z = exo.obstacle_crossing_with_force_field(st_hip,st_knee,sw_hip,sw_knee)
    
    relative_ankle_px,relative_ankle_pz = exo.relative_ankle_distance(st_hip,st_knee,sw_hip,sw_knee)
    updated_ankle_px,updated_ankle_pz = exo.relative_ankle_distance(st_hip,st_knee,updated_sw_h,updated_sw_k)
    
    # plot raw\updated joint angle, repulsive force, foot trajectory
    plt.figure(1)
    plt.subplot(221)
    plt.plot(sw_hip,'r--')
    plt.plot(sw_knee,'b--')
    plt.plot(updated_sw_h,'r')
    plt.plot(updated_sw_k,'b')
    plt.legend(('dmp swing hip','dmp swing knee','updated swing hip','updated swing knee'))
    plt.ylabel('joint angle/deg')
    
    plt.subplot(222)
    plt.plot(raw_st_hip,'r--')
    plt.plot(raw_st_knee,'b--')
    plt.plot(st_hip,'r')
    plt.plot(st_knee,'b')
    plt.legend(('raw stance hip','raw stance knee','dmp stance hip','dmp stance knee'))
    plt.ylabel('joint angle/deg')

    plt.subplot(223)
    plt.plot(relative_ankle_px,relative_ankle_pz,'b--',lw=2.0) # end effector trajectory
    plt.plot(updated_ankle_px,updated_ankle_pz,'c-',lw=2.0)
    plt.plot(obstacle_px,obstacle_pz,'r',lw=2.0)
    plt.legend(('original end effector','post end effector','obstacle'))
    plt.xlabel('px/m')
    plt.ylabel('pz/m')
    
    plt.subplot(224)
    plt.plot(force_x)
    plt.plot(force_z)
    plt.legend(('$\Delta$x','$\Delta$z'))
    plt.ylabel('adjustment/m')
    
    
    # plot EXO simulation animation
    plt.figure(2)
    exo.plot_one_step(st_hip,st_knee,updated_sw_h,updated_sw_k,terrain_type)
    plt.show()
elif terrain_type == "slope":
    
    step_length = 0.375
    # first step
    y = y_des[:,0] + [0,0,0,0]
    new_scale = np.ones(y.shape[0]) + [0,0,0,0]
    goal_offset = np.zeros(y.shape[0]) + [0,0,0,0]
    
    # Adjust the ending joint angle for ramp walking
    target_angle = JointAngleForRamp(y_des[:,-1],thigh_length,shank_length,slope,step_length)
    goal_offset = target_angle - y_des[:,-1]

    # Hip dmp unit needs the extral scale to correct the affect from the goal offset
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    new_scale[3] = (1.0 - target_angle[3] / 60.0) * step_length / 0.75 # for first method to deal with slope
    # new_scale[3] = (dmp_gait.goal[1]+goal_offset[1])/(dmp_gait.goal[1]) # for second method(rotate body with slope angle)
    track,track_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)
    
    # second step
    y = np.array([track[1,-1],track[0,-1],track[3,-1],track[2,-1]])
    target_angle = JointAngleForRamp(y_des[:,-1],thigh_length,shank_length,slope,step_length)
    goal_offset = target_angle - y_des[:,-1]
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    new_scale[3] = (1.0 - target_angle[3] / 60.0)  * step_length / 0.75
    # new_scale[3] = (dmp_gait.goal[1]+goal_offset[1])/(dmp_gait.goal[1]) # for second method(rotate body with slope angle)
    track2,track2_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)
    
    
    # plot first step in slope
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
            
    # plot second step in slope
    st_hip=track2[0,:]
    sw_hip=track2[1,:]
    st_knee=track2[2,:]
    sw_knee=track2[3,:]
    f1, axs1 = plt.subplots(2,2)
    axs1[0][0].plot(y_des_time, raw_st_hip, label='raw st_hip')
    axs1[0][0].plot(track2_time, st_hip, label='dmp st_hip')
    axs1[1][0].plot(y_des_time, raw_st_knee, label='raw st_knee')
    axs1[1][0].plot(track2_time, st_knee, label='dmp st_knee')

    axs1[0][1].plot(y_des_time, raw_sw_hip, label='raw sw_hip')
    axs1[0][1].plot(track2_time, sw_hip, label='dmp sw_hip')
    axs1[1][1].plot(y_des_time, raw_sw_knee, label='raw sw_knee')
    axs1[1][1].plot(track2_time, sw_knee, label='dmp sw_knee')
    for ax in axs1:
        for a in ax:
            a.legend()
    
    
    # plot EXO simulation animation
    # plt.figure(3)
    # lh = track[0,:]
    # lk = track[2,:]
    # rh = track[1,:]
    # rk = track[3,:]
    # exo.plot_one_step(lh,lk,rh,rk,terrain_type)
    
    # lh = track2[1,:]
    # lk = track2[3,:]
    # rh = track2[0,:]
    # rk = track2[2,:]
    # exo.update_stace_leg(False)
    # exo.plot_one_step(lh,lk,rh,rk,terrain_type)
    # plt.show()
    
    # save to gif
    plt.figure(3)
    filename = 'fig.png'
    with imageio.get_writer('slope_walking.gif', mode='I') as writer:
        lh = track[0,:]
        lk = track[2,:]
        rh = track[1,:]
        rk = track[3,:]
        for i in range(num//5):
            j = i * 5
            exo.plot_exo(lh[j],lk[j],rh[j],rk[j],terrain_type)
            plt.savefig(filename)
            plt.close
            image = imageio.imread(filename)
            writer.append_data(image)
        lh = track2[1,:]
        lk = track2[3,:]
        rh = track2[0,:]
        rk = track2[2,:]
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
else:
    y = y_des[:,0] + [0,0,0,0]
    new_scale = np.ones(y.shape[0]) + [0,0,0,0]
    goal_offset = np.zeros(y.shape[0]) + [0,0,0,0]
    
    # For leve ground walking, it it to adjust the ending position of swing and stance hip joint for step length adjustment
    step_length = 0.5
    tmp_st_hip, tmp_sw_hip = CorrectHipForStepLength(y_des[:,-1],thigh_length,shank_length,step_length)
    goal_offset[0] = tmp_st_hip - y_des[0,-1]
    goal_offset[1] = tmp_sw_hip - y_des[1,-1]
    
    # Hip dmp unit needs the extral scale to correct the affect from the goal offset
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    new_scale[3] = (dmp_gait.goal[1]+goal_offset[1])/(dmp_gait.goal[1])
    track,track_time = dmp_gait_generation(dmp_gait,num_steps=500,y0=y,new_scale=new_scale,goal_offset=goal_offset)
    
    # second step
    y = np.array([track[1,-1],track[0,-1],track[3,-1],track[2,-1]])
    tmp_st_hip, tmp_sw_hip = CorrectHipForStepLength(y_des[:,-1],thigh_length,shank_length,step_length)
    goal_offset[0] = tmp_st_hip - y_des[0,-1]
    goal_offset[1] = tmp_sw_hip - y_des[1,-1]
    new_scale[0] = (dmp_gait.goal[0]+goal_offset[0]-y[0])/(dmp_gait.goal[0]-dmp_gait.y0[0])
    new_scale[1] = (dmp_gait.goal[1]+goal_offset[1]-y[1])/(dmp_gait.goal[1]-dmp_gait.y0[1])
    new_scale[3] = (dmp_gait.goal[1]+goal_offset[1])/(dmp_gait.goal[1])
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
    


# # save to gif
# exo.reset()
# plt.figure(3)
# filename = 'fig.png'
# with imageio.get_writer('raw_gait.gif', mode='I') as writer:
#     for i in range(num//5):
#         j = i * 5
#         exo.plot_exo(lh[j],lk[j],rh[j],rk[j],"obstacle")
        
#         plt.savefig(filename)
#         plt.close
#         image = imageio.imread(filename)
#         writer.append_data(image)
        
# os.remove(filename)       
# plt.show()
