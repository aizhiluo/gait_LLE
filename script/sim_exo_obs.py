from sim import EXO_OBS
import numpy as np
from time import sleep
import matplotlib.pyplot as plt
import imageio
import os

from copy import deepcopy
from dmp import DMP
from utils import Kinematics, InverseKinematics, JacobianMatrix

def read_gait_data_txt():
    path = "D:/MyFolder/code/EXO_ROS/Exoskeleton_WP3/machine_learning/gait/data/"
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
    txt_file.close()
    
    st_hip = np.array(st_hip_list)
    st_knee = np.array(st_knee_list)
    sw_hip = np.array(sw_hip_list)
    sw_knee = np.array(sw_knee_list)
    
    return st_hip, st_knee, sw_hip, sw_knee

def dmp_gait_generation(st_hip,st_knee,sw_hip,sw_knee,time_interval=0.001):
    y_des = np.array([st_hip, sw_hip, st_knee, sw_knee])
    y_des_time = np.arange(0, y_des.shape[1])*time_interval
    
    # create DMP object
    ay = np.ones(4) * 50
    by = ay / 4
    dmp_gait = DMP(y_des, y_des_time, ay=ay, by=by, n_bfs=250, dt=time_interval, isz=True)
    _, dmp_time_left_swing = dmp_gait.imitate_path()
    
    traj_num = y_des.shape[0]
    num_steps = y_des.shape[1]
    track = np.zeros((traj_num, num_steps))
    dmp_time = np.arange(num_steps) * time_interval
    
    # the goal_offset, scale, and initial position for the generated trajectory
    goal_offset = np.array([0,0,0,0])
    new_scale = np.ones(num_steps)
    y = y_des[:,0]
    dy = np.zeros(traj_num)
    force = np.zeros(traj_num)
    
     # Generate target trajectory using DMP step by step
    for i in range(num_steps):            
        gait_phase = float(i) / num_steps
        y, dy, ddy = dmp_gait.step_real(gait_phase, y, dy, scale=new_scale, goal_offset=goal_offset)
        
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
        
    dmp_st_hip = track[0,:]
    dmp_sw_hip = track[1,:]
    dmp_st_knee = track[2,:]
    dmp_sw_knee = track[3,:]
    
    return dmp_st_hip, dmp_st_knee, dmp_sw_hip, dmp_sw_knee
    
    
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
    alpha = 1.0
    beta = 63.25
    k = 10.0
    
    # potential fields repulsive force formula parameters
    radius = max(obs_h,obs_w) / np.sqrt(2)
    D = 0.05 # tolent space
    k = alpha / (radius + D)
    beta = np.sqrt(k/(radius * D))
    
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
st_hip, st_knee, sw_hip, sw_knee=read_gait_data_txt()
st_hip, st_knee, sw_hip, sw_knee=dmp_gait_generation(st_hip, st_knee, sw_hip, sw_knee)

num = st_hip.shape[0]

# initial exo and obstacle
thigh_length = 0.44
shank_length = 0.565
obstacle_dist = 0.15 # assume obstacle locates in x-axis z = 0
obstacle_width = 0.1
obstacle_height = 0.1
exo = EXO_OBS(obstacle_dist,obstacle_height,obstacle_width,thigh_length,shank_length)

# first obstacle step with left swing
lh = st_hip
lk = st_knee
rh = sw_hip
rk = sw_knee
exo.update_stace_leg(True)

# relative distance between swing and stance foot
hip_angle = np.array([st_hip,sw_hip])
knee_angle = np.array([st_knee,sw_knee])
foot_px,foot_pz = Kinematics(hip_angle,knee_angle,thigh_length,shank_length)
px = foot_px[1,:] - foot_px[0,:]
pz = foot_pz[1,:] - foot_pz[0,:]

# adjust the swing foot position according to the relative distance
force_x,force_z = RepusiveForceFile(px,pz,obstacle_dist,obstacle_width,obstacle_height)
post_sw_px = foot_px[1,:] + force_x
post_sw_pz = foot_pz[1,:] + force_z
updated_sw_h,updated_sw_k = InverseKinematics(np.array([post_sw_px,post_sw_pz]),thigh_length,shank_length)

# plot one step
plt.figure()
exo.plot_one_step(True,lh,lk,updated_sw_h,updated_sw_k)
plt.show()

plt.figure(2)

plt.subplot(221)
plt.plot(hip_angle[1,:],'r--')
plt.plot(knee_angle[1,:],'b--')
plt.plot(updated_sw_h,'r')
plt.plot(updated_sw_k,'b')
plt.legend(('swing hip','swing knee','post swing hip','post swing knee'))
plt.ylabel('joint angle/deg')

plt.subplot(222)
plt.plot(force_x)
plt.plot(force_z)
plt.legend(('$\Delta$x','$\Delta$z'))
plt.ylabel('adjustment/m')

plt.subplot(223)
plt.plot(px,pz,'b--',lw=2.0) # end effector trajectory
plt.plot(post_sw_px-foot_px[0,:],post_sw_pz-foot_pz[0,:],'c-',lw=2.0)
plt.legend(('original end effector','post end effector','obstacle'))
plt.xlabel('px/m')
plt.ylabel('pz/m')

plt.show()

# # save to gif
# num = st_hip.shape[0]
# plt.figure()
# filename = 'fig.png'
# with imageio.get_writer('raw_gait.gif', mode='I') as writer:
#     for i in range(num//5):
#         j = i * 5
#         exo.plot_exo(True,lh[j],lk[j],rh[j],rk[j])
        
#         plt.savefig(filename)
#         plt.close
#         image = imageio.imread(filename)
#         writer.append_data(image)
        
# os.remove(filename)       
# plt.show()
