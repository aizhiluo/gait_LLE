from sim import EXO_OBS
import numpy as np
from time import sleep
import matplotlib.pyplot as plt

from utils import Kinematics, InverseKinematics

def read_gait_data(file_path):
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
            force_x += force * (px-corner_x[i])/np.sqrt(dist) * x_force_scale
            force_z += force * (pz-corner_z[i])/np.sqrt(dist)
    
    return force_x, (force_z + abs(force_z))/2.0


# load gait data
path = "D:/MyFolder/code/EXO_ROS/Exoskeleton_WP3/machine_learning/gait/data/"
file_name = "SN_all_subj_obstacle_first"
file_path = path + file_name + ".txt"
st_hip, st_knee, sw_hip, sw_knee=read_gait_data(file_path)
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
rh,rk = InverseKinematics(np.array([post_sw_px,post_sw_pz]),thigh_length,shank_length)


for i in range(num//5):
    j = i * 5
    plt.figure(1)
    exo.plot_exo(True,lh[j],lk[j],rh[j],rk[j])


# lh = sw_hip
# lk = sw_knee
# rh = st_hip
# rk = st_knee
# exo.update_stace_leg(False)
# for i in range(num):
#     plt.figure(1)
#     exo.plot_exo(False,lh[i],lk[i],rh[i],rk[i])
    
while(1):
   sleep(0.05) 
