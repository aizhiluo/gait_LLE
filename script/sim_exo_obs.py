from sim import EXO_OBS
import numpy as np
from time import sleep

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
    
    st_hip = np.array(st_hip_list)
    st_knee = np.array(st_knee_list)
    sw_hip = np.array(sw_hip_list)
    sw_knee = np.array(sw_knee_list)
    
    return st_hip, st_knee, sw_hip, sw_knee

path = "D:/MyFolder/code/EXO_ROS/Exoskeleton_WP3/machine_learning/gait/data/"
file_name = "SN_all_subj_obstacle_first"
file_path = path + file_name + ".txt"
st_hip, st_knee, sw_hip, sw_knee=read_gait_data(file_path)

exo = EXO_OBS(0.56,0.44)
num = st_hip.shape[0]
for i in range(num):
    lh = st_hip[i]
    lk = st_knee[i]
    rh = sw_hip[i]
    rk = sw_knee[i]
    exo.plot_exo(True,lh,lk,rh,rk)

lh = sw_hip
lk = sw_knee
rh = st_hip
rk = st_knee
exo.update_stace_leg(False,lh[0],lk[0],rh[0],rk[0])
for i in range(num):
    exo.plot_exo(True,lh[i],lk[i],rh[i],rk[i])
    
while(1):
   sleep(0.05) 
