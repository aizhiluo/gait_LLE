from pathlib import Path

from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from dmp import DMP
from utils import bodyM, ideal_body_p, load_gait_data, r_2_d, slice_gait_phase, Kinematics
import csv
import sys


''' Read data from MoCop csv file '''

def ReadDataFromCSV(file_path):
    # Load gait data (consist of two steps)
    data = load_gait_data(file_path)
    # Seperate a full gait by heel strike.
    idx = np.squeeze(np.argwhere(data["l_hs"] == True))
    left_swing_data = slice_gait_phase(data, 0, idx + 1)
    right_swing_data = slice_gait_phase(data, idx + 1, len(data["l_hs"]) + 1)

    left_swing_traj = { 
        "xf_r": left_swing_data["r_ankle_global_x"],
        "xf_l": left_swing_data["l_ankle_global_x"],
        "zf_r": left_swing_data["r_ankle_global_z"],
        "zf_l": left_swing_data["l_ankle_global_z"],
    }

    right_swing_traj = { 
        "xf_r": right_swing_data["r_ankle_global_x"],
        "xf_l": right_swing_data["l_ankle_global_x"],
        "zf_r": right_swing_data["r_ankle_global_z"],
        "zf_l": right_swing_data["l_ankle_global_z"],
    }
    left_swing_time = left_swing_data["time"]
    right_swing_time = right_swing_data["time"] - right_swing_data["time"][0] #reset to start from 0

    #Get the hip, knee and ankle position in local coord (zero at hip)
    right_ankle_left_swing = np.array([left_swing_data["r_ankle_local_x"], left_swing_data["r_ankle_local_y"], left_swing_data["r_ankle_local_z"]])
    left_ankle_left_swing = np.array([left_swing_data["l_ankle_local_x"], left_swing_data["l_ankle_local_y"], left_swing_data["l_ankle_local_z"]])
    right_knee_left_swing = np.array([left_swing_data["r_knee_local_x"], left_swing_data["r_knee_local_y"], left_swing_data["r_knee_local_z"]])
    left_knee_left_swing = np.array([left_swing_data["l_knee_local_x"], left_swing_data["l_knee_local_y"], left_swing_data["l_knee_local_z"]])
    right_hip_left_swing = np.array([left_swing_data["r_hip_local_x"], left_swing_data["r_hip_local_y"], left_swing_data["r_hip_local_z"]])
    left_hip_left_swing = np.array([left_swing_data["l_hip_local_x"], left_swing_data["l_hip_local_y"], left_swing_data["l_hip_local_z"]])
    right_ankle_right_swing = np.array([right_swing_data["r_ankle_local_x"], right_swing_data["r_ankle_local_y"], right_swing_data["r_ankle_local_z"]])
    left_ankle_right_swing = np.array([right_swing_data["l_ankle_local_x"], right_swing_data["l_ankle_local_y"], right_swing_data["l_ankle_local_z"]])
    right_knee_right_swing = np.array([right_swing_data["r_knee_local_x"], right_swing_data["r_knee_local_y"], right_swing_data["r_knee_local_z"]])
    left_knee_right_swing = np.array([right_swing_data["l_knee_local_x"], right_swing_data["l_knee_local_y"], right_swing_data["l_knee_local_z"]])
    right_hip_right_swing = np.array([right_swing_data["r_hip_local_x"], right_swing_data["r_hip_local_y"], right_swing_data["r_hip_local_z"]])
    left_hip_right_swing = np.array([right_swing_data["l_hip_local_x"], right_swing_data["l_hip_local_y"], right_swing_data["l_hip_local_z"]])

    #Plot the knees and ankles path
    f, axs = plt.subplots(2,2)
    axs[0][0].plot(right_knee_left_swing[0], right_knee_left_swing[2], label='Right knee left swing')
    axs[0][0].plot(left_knee_left_swing[0], left_knee_left_swing[2], label='Left knee left swing')
    axs[1][0].plot(right_ankle_left_swing[0], right_ankle_left_swing[2], label='Right ankle left swing')
    axs[1][0].plot(left_ankle_left_swing[0], left_ankle_left_swing[2], label='Left ankle left swing')
    axs[0][1].plot(right_knee_right_swing[0], right_knee_right_swing[2], label='Right knee right swing')
    axs[0][1].plot(left_knee_right_swing[0], left_knee_right_swing[2], label='Left knee right swing')
    axs[1][1].plot(right_ankle_right_swing[0], right_ankle_right_swing[2], label='Right ankle right swing')
    axs[1][1].plot(left_ankle_right_swing[0], left_ankle_right_swing[2], label='Left ankle right swing')
    for i in range(2):
        for j in range(2):
            axs[i][j].legend()
            
    ''' Find joint angles '''
    left_thigh_left_swing = np.array(left_knee_left_swing - left_hip_left_swing).T
    right_thigh_left_swing = np.array(right_knee_left_swing - right_hip_left_swing).T
    left_thigh_right_swing = np.array(left_knee_right_swing - left_hip_right_swing).T
    right_thigh_right_swing = np.array(right_knee_right_swing - right_hip_right_swing).T
    left_shank_left_swing = np.array(left_ankle_left_swing - left_knee_left_swing).T
    right_shank_left_swing = np.array(right_ankle_left_swing - right_knee_left_swing).T
    left_shank_right_swing = np.array(left_ankle_right_swing - left_knee_right_swing).T
    right_shank_right_swing = np.array(right_ankle_right_swing - right_knee_right_swing).T
    left_hip_angle_left_swing = np.arctan2(left_thigh_left_swing[:, 0], -left_thigh_left_swing[:, 2]) * 180 / np.pi 
    right_hip_angle_left_swing = np.arctan2(right_thigh_left_swing[:, 0], -right_thigh_left_swing[:, 2]) * 180 / np.pi
    left_hip_angle_right_swing = np.arctan2(left_thigh_right_swing[:, 0], -left_thigh_right_swing[:, 2]) * 180 / np.pi
    right_hip_angle_right_swing = np.arctan2(right_thigh_right_swing[:, 0], -right_thigh_right_swing[:, 2]) * 180 / np.pi
    left_knee_angle_left_swing = -(np.arctan2(left_shank_left_swing[:, 0], -left_shank_left_swing[:, 2]) * 180 / np.pi - left_hip_angle_left_swing)
    right_knee_angle_left_swing = -(np.arctan2(right_shank_left_swing[:, 0], -right_shank_left_swing[:, 2]) * 180 / np.pi - right_hip_angle_left_swing)
    left_knee_angle_right_swing = -(np.arctan2(left_shank_right_swing[:, 0], -left_shank_right_swing[:, 2]) * 180 / np.pi - left_hip_angle_right_swing)
    right_knee_angle_right_swing = -(np.arctan2(right_shank_right_swing[:, 0], -right_shank_right_swing[:, 2]) * 180 / np.pi - right_hip_angle_right_swing)

    return left_hip_angle_left_swing,right_hip_angle_left_swing,left_hip_angle_right_swing,right_hip_angle_right_swing,left_knee_angle_left_swing,right_knee_angle_left_swing,left_knee_angle_right_swing,right_knee_angle_right_swing,left_swing_time,right_swing_time

data_file_path = "./data/mocap/SN002_0028_10m_01.csv"
lhip_lsw,rhip_lsw,lhip_rsw,rhip_rsw,lknee_lsw,rknee_lsw,lknee_rsw,rknee_rsw,left_swing_time,right_swing_time = ReadDataFromCSV(data_file_path)
y_des_left_swing = np.array([lhip_lsw, lknee_lsw, rhip_lsw, rknee_lsw]) # sw_hip, sw_knee, st_hip, st_knee
print(y_des_left_swing.shape)

ay = np.ones(4) * 50
by = ay / 4
dmp_150_kernel = DMP(y_des_left_swing, left_swing_time, ay=ay, by=by, n_bfs=150, dt=0.001, isz=True)
_, dmp_time_left_swing = dmp_150_kernel.imitate_path()

dmp_50_kernel = DMP(y_des_left_swing, left_swing_time, ay=ay, by=by, n_bfs=50, dt=0.001, isz=True)
_, dmp_time_50_kernel = dmp_50_kernel.imitate_path()

dmp_500_kernel = DMP(y_des_left_swing, left_swing_time, ay=ay, by=by, n_bfs=500, dt=0.001, isz=True)
_, dmp_time_500_kernel = dmp_500_kernel.imitate_path()

dmp_150_kernel.save_para(f"./data/dmp_para/left_swing")


# Verify dmp fitting.
goal, tau = 1.0, 1.0
track_150_kernel, track_res_vel_left_swing, _, time_left_swing = dmp_150_kernel.rollout(goal=goal, tau=tau)
track_50_kernel, _, _, _ = dmp_50_kernel.rollout(goal=goal, tau=tau)
track_500_kernel, _, _, _ = dmp_500_kernel.rollout(goal=goal, tau=tau)

# # Save the trajectories
# with open("./dmp_temp/baseline_trajectory.txt", mode="w", newline="") as data_file:
#     writer = csv.writer(data_file, delimiter=",")
#     data_file.write(f"#size: {y_des_left_swing.shape[1]} 4\n")
#     data_file.write(f"#sw_hip, sw_knee, st_hip, st_knee\n")
#     for i in range(y_des_left_swing.shape[1]):
#         writer.writerow(y_des_left_swing[:, i])
        
# with open("./dmp_temp/150_kernel_trajectory.txt", mode="w", newline="") as data_file:
#     writer = csv.writer(data_file, delimiter=",")
#     data_file.write(f"#size: {track_150_kernel.shape[0]} 4\n")
#     data_file.write(f"#sw_hip, sw_knee, st_hip, st_knee\n")
#     for i in range(track_150_kernel.shape[0]):
#         writer.writerow(track_150_kernel[i, :])

# with open("./dmp_temp/50_kernel_trajectory.txt", mode="w", newline="") as data_file:
#     writer = csv.writer(data_file, delimiter=",")
#     data_file.write(f"#size: {track_50_kernel.shape[0]} 4\n")
#     data_file.write(f"#sw_hip, sw_knee, st_hip, st_knee\n")
#     for i in range(track_50_kernel.shape[0]):
#         writer.writerow(track_50_kernel[i, :])
        
# with open("./dmp_temp/500_kernel_trajectory.txt", mode="w", newline="") as data_file:
#     writer = csv.writer(data_file, delimiter=",")
#     data_file.write(f"#size: {track_500_kernel.shape[0]} 4\n")
#     data_file.write(f"#sw_hip, sw_knee, st_hip, st_knee\n")
#     for i in range(track_500_kernel.shape[0]):
#         writer.writerow(track_500_kernel[i, :])

# fig, axs = plt.subplots(2,2,sharey="row")
# axs[0,0].set_title("DMP Fitting 150 Kernel - Hip")
# axs[0,0].set_xlabel("Time (sec)")
# axs[0,0].set_ylabel("Degree")
# axs[0,0].plot(left_swing_time, rhip_lsw, lw=5, label="right hip joint ideal")
# axs[0,0].plot(left_swing_time, lhip_lsw, lw=5, label="left hip joint ideal")
# axs[0,0].plot(time_left_swing, track_150_kernel[:, 2], lw=1, label="right hip joint dmp")
# axs[0,0].plot(time_left_swing, track_150_kernel[:, 0], lw=1, label="left hip joint dmp")
# axs[1,0].set_title("DMP Fitting 150 Kernel - Knee")
# axs[1,0].set_xlabel("Time (sec)")
# axs[1,0].set_ylabel("Degree")
# axs[1,0].plot(left_swing_time, rknee_lsw, lw=5, label="right knee joint ideal")
# axs[1,0].plot(left_swing_time, lknee_lsw, lw=5, label="left knee joint ideal")
# axs[1,0].plot(time_left_swing, track_150_kernel[:, 3], lw=1, label="right knee joint dmp")
# axs[1,0].plot(time_left_swing, track_150_kernel[:, 1], lw=1, label="left knee joint dmp")
# axs[0,1].set_title("DMP Fitting 50 Kernel - Hip")
# axs[0,1].set_xlabel("Time (sec)")
# axs[0,1].set_ylabel("Degree")
# axs[0,1].plot(right_swing_time, rhip_rsw, lw=5, label="right hip joint ideal")
# axs[0,1].plot(right_swing_time, lhip_rsw, lw=5, label="left hip joint ideal")
# axs[0,1].plot(time_left_swing, track_50_kernel[:, 2], lw=1, label="right hip joint dmp")
# axs[0,1].plot(time_left_swing, track_50_kernel[:, 0], lw=1, label="left hip joint dmp")
# axs[1,1].set_title("DMP Fitting 50 Kernel - Knee")
# axs[1,1].set_xlabel("Time (sec)")
# axs[1,1].set_ylabel("Degree")
# axs[1,1].plot(right_swing_time, rknee_rsw, lw=5, label="right knee joint ideal")
# axs[1,1].plot(right_swing_time, lknee_rsw, lw=5, label="left knee joint ideal")
# axs[1,1].plot(time_left_swing, track_50_kernel[:, 3], lw=1, label="right knee joint dmp")
# axs[1,1].plot(time_left_swing, track_50_kernel[:, 1], lw=1, label="left knee joint dmp")
# for ax in axs:
#     for a in ax:
#         a.legend()
# plt.show()

# Generate trajectory with y0, scale, goal offset
traj_num = track_50_kernel.shape[1]
num_steps = track_50_kernel.shape[0]
track = np.zeros((num_steps,traj_num))
dmp_time = np.arange(num_steps) * 0.001

# the goal_offset, scale, and initial position for the generated trajectory
goal_offset = np.array([0,0,0,0])
new_scale = np.ones(num_steps)*1.2
y = y_des_left_swing[:,0]
dy = np.zeros(traj_num)

    
for i in range(num_steps):
    gait_phase = float(i) / num_steps
    
    y, dy, ddy = dmp_150_kernel.step_real(gait_phase, y, dy, scale=new_scale, goal_offset=goal_offset, tau=1.0, extra_force=0.0)
    track[i,:] = deepcopy(y)

dmp_sw_hip = track[:,0]
dmp_sw_knee = track[:,1]
dmp_st_hip = track[:,2]
dmp_st_knee = track[:,3]

# with open("./dmp_temp/offset_8.0_trajectory.txt", mode="w", newline="") as data_file:
#     writer = csv.writer(data_file, delimiter=",")
#     data_file.write(f"#size: {track.shape[0]} 4\n")
#     data_file.write(f"#sw_hip, sw_knee, st_hip, st_knee\n")
#     for i in range(track.shape[0]):
#         writer.writerow(track[i,:])


# endeffector position
shank_length = 0.565
thigh_length = 0.44
px,pz = Kinematics(lhip_lsw,lknee_lsw,thigh_length,shank_length)
dmp_px,dmp_pz = Kinematics(dmp_sw_hip,dmp_sw_knee,thigh_length,shank_length)

# Plot original and generated trajectories
f, axs = plt.subplots(1,2)
axs[0].plot(left_swing_time, lhip_lsw, label='sw_hip')
axs[0].plot(dmp_time, dmp_sw_hip, label='dmp sw_hip')
axs[1].plot(left_swing_time, lknee_lsw, label='sw_knee')
axs[1].plot(dmp_time, dmp_sw_knee, label='dmp sw_knee')
for ax in axs:
        ax.legend()
        
f, axs = plt.subplots()         
axs.plot(px,pz)
axs.plot(dmp_px,dmp_pz)

plt.show()

