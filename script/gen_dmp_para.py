from pathlib import Path

from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from movement import DMP
from utils import bodyM, ideal_body_p, load_gait_data, r_2_d, slice_gait_phase
import csv
import sys

use_joint_angle = True
is_data_from_mocop = True

''' Read data from MoCop csv file '''

def ReadDataFromText(file_path):
    txt_file = open(file_path, 'r')
    # read data starting from the third line
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
    
    num = st_hip.shape[0]
    times = np.arange(0, st_hip.shape[0])*0.001
    
    return st_hip, st_knee, sw_hip, sw_knee, times
    
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
    

''' Animation of the gait patterns '''

# fig, ax = plt.subplots()
# ax.set_xlim(-1000, 1000)
# ax.set_ylim(-1000, 1000)
# ax.axvline(x=0, color='k', lw=1.0)
# ax.axhline(y=0, color='k', lw=1.0)
# ax.set_xlabel('x')
# ax.set_ylabel('z')
# ax.set_title("Red: left leg, Blue: right leg")

# l_r_1, = ax.plot([], [], c='b') #plot the leg
# l_l_1, = ax.plot([], [], c='r') #plot the leg
# l_r_2, = ax.plot([], [], c='b') #plot the leg
# l_l_2, = ax.plot([], [], c='r') #plot the leg
# import matplotlib.animation as animation

# def updateAnim(i, l_r_1, l_l_1, l_r_2, l_l_2):
#     j_r_1 = [right_hip_left_swing[0][i], right_knee_left_swing[0][i], right_ankle_left_swing[0][i]], [right_hip_left_swing[2][i], right_knee_left_swing[2][i], right_ankle_left_swing[2][i]]
#     j_l_1 = [left_hip_left_swing[0][i], left_knee_left_swing[0][i], left_ankle_left_swing[0][i]], [left_hip_left_swing[2][i], left_knee_left_swing[2][i], left_ankle_left_swing[2][i]]
#     j_r_2 = [right_hip_right_swing[0][i], right_knee_right_swing[0][i], right_ankle_right_swing[0][i]], [right_hip_right_swing[2][i], right_knee_right_swing[2][i], right_ankle_right_swing[2][i]]
#     j_l_2 = [left_hip_right_swing[0][i], left_knee_right_swing[0][i], left_ankle_right_swing[0][i]], [left_hip_right_swing[2][i], left_knee_right_swing[2][i], left_ankle_right_swing[2][i]]
#     l_r_1.set_data(j_r_1[0] - np.ones(3)*400, j_r_1[1]) #Plot the leg length
#     l_l_1.set_data(j_l_1[0] - np.ones(3)*400, j_l_1[1]) #Plot the leg length
#     l_r_2.set_data(j_r_2[0] + np.ones(3)*400, j_r_2[1]) #Plot the leg length
#     l_l_2.set_data(j_l_2[0] + np.ones(3)*400, j_l_2[1]) #Plot the leg length
    
#     return l_r_1, l_l_1, l_r_2, l_l_2

# traj_len = np.min([right_hip_left_swing[0].shape[0], right_hip_right_swing[0].shape[0]])
# anim = animation.FuncAnimation(fig, updateAnim, traj_len,
#     fargs=(l_r_1, l_l_1, l_r_2, l_l_2),
#     interval=10, repeat_delay=1000, blit=False)

# plt.show()
# sys.exit()


# ''' From Free Exo '''
# left_sw_df = pd.read_excel("gait/Extracted_Swing_Data_Combined.xlsx", sheet_name="LC-normal_QS-LeftSw", index_col=0)
# right_sw_df = pd.read_excel("gait/Extracted_Swing_Data_Combined.xlsx", sheet_name="LC-normal_QS-RightSw", index_col=0)
# # left_hip_angle_left_swing = left_sw_df["LHip"].values
# # left_knee_angle_left_swing = (left_sw_df["LKnee"].values - left_sw_df["LKnee"].values[0]) * 1 + left_sw_df["LKnee"].values[0]
# # right_hip_angle_left_swing = (left_sw_df["RHip"].values - left_sw_df["RHip"].values[0]) * 1.7 + left_sw_df["RHip"].values[0]
# # right_knee_angle_left_swing = (left_sw_df["RKnee"].values - left_sw_df["RKnee"].values[0] - 11) * 1 + left_sw_df["RKnee"].values[0]

# # left_hip_angle_right_swing = (right_sw_df["LHip"].values - right_sw_df["LHip"].values[0]) * 1.5 + right_sw_df["LHip"].values[0]
# # left_knee_angle_right_swing = (right_sw_df["LKnee"].values - right_sw_df["LKnee"].values[0]) * 1 + right_sw_df["LKnee"].values[0]
# # right_hip_angle_right_swing = right_sw_df["RHip"].values
# # right_knee_angle_right_swing = (right_sw_df["RKnee"].values - right_sw_df["RKnee"].values[0]) * 1 + right_sw_df["RKnee"].values[0]

# left_hip_angle_left_swing = left_sw_df["LHip"].values
# left_knee_angle_left_swing = (left_sw_df["LKnee"].values - left_sw_df["LKnee"].values[0]) * 1 + left_sw_df["LKnee"].values[0]
# right_hip_angle_left_swing = (left_sw_df["RHip"].values - left_sw_df["RHip"].values[0]) * 1 + left_sw_df["RHip"].values[0]
# right_knee_angle_left_swing = (left_sw_df["RKnee"].values - left_sw_df["RKnee"].values[0]) * 1 + left_sw_df["RKnee"].values[0]

# left_hip_angle_right_swing = (right_sw_df["LHip"].values - right_sw_df["LHip"].values[0]) * 1 + right_sw_df["LHip"].values[0]
# left_knee_angle_right_swing = (right_sw_df["LKnee"].values - right_sw_df["LKnee"].values[0]) * 1 + right_sw_df["LKnee"].values[0]
# right_hip_angle_right_swing = right_sw_df["RHip"].values
# right_knee_angle_right_swing = (right_sw_df["RKnee"].values - right_sw_df["RKnee"].values[0]) * 1 + right_sw_df["RKnee"].values[0]
# left_swing_time = np.linspace(0, 0.45, left_sw_df.shape[0])
# right_swing_time = np.linspace(0, 0.45, right_sw_df.shape[0])
# ''' From Free Exo ENDS '''

# print(np.mean(np.linalg.norm(left_thigh_left_swing, axis=1)), np.mean(np.linalg.norm(right_thigh_left_swing, axis=1)),
#     np.mean(np.linalg.norm(left_shank_left_swing, axis=1)), np.mean(np.linalg.norm(right_shank_left_swing, axis=1)))
# sys.exit()

# #Plot joint angles
# fig, axs = plt.subplots(2, 2, sharex=True, sharey=True)
# axs[0,0].plot(left_hip_angle_left_swing, label='left_hip_angle_left_swing')
# axs[0,0].plot(right_hip_angle_left_swing, label='right_hip_angle_left_swing')
# axs[0,1].plot(left_knee_angle_left_swing, label='left_knee_angle_left_swing')
# axs[0,1].plot(right_knee_angle_left_swing, label='right_knee_angle_left_swing')
# axs[1,0].plot(left_hip_angle_right_swing, label='left_hip_angle_right_swing')
# axs[1,0].plot(right_hip_angle_right_swing, label='right_hip_angle_right_swing')
# axs[1,1].plot(left_knee_angle_right_swing, label='left_knee_angle_right_swing')
# axs[1,1].plot(right_knee_angle_right_swing, label='right_knee_angle_right_swing')
# # axs[0,0].plot(left_hip_angle_left_swing, left_knee_angle_left_swing, label='L1')
# # axs[0,1].plot(right_hip_angle_left_swing, right_knee_angle_left_swing, label='R1')
# # axs[1,0].plot(left_hip_angle_right_swing, left_knee_angle_right_swing, label='L2')
# # axs[1,1].plot(right_hip_angle_right_swing, right_knee_angle_right_swing, label='R2')
# for ax in axs:
#     for a in ax:
#         a.legend()
#         a.axhline(y=0, c='k')

# plt.show()
# sys.exit()

''' Fitting DMP '''
if is_data_from_mocop is True:
    data_file_path = "./machine_learning/gait/SN002_0028_10m_01.csv"
    lhip_lsw,rhip_lsw,lhip_rsw,rhip_rsw,lknee_lsw,rknee_lsw,lknee_rsw,rknee_rsw,left_swing_time,right_swing_time = ReadDataFromCSV(data_file_path)
    y_des_left_swing = np.array([rhip_lsw, lhip_lsw, rknee_lsw, lknee_lsw]) # st_hip, sw_hip, st_knee, sw_knee
    y_des_right_swing = np.array([lhip_rsw, rhip_rsw, lknee_rsw, rknee_rsw])
    print(y_des_right_swing.shape)

    ay = np.ones(4) * 50
    by = ay / 4
    dmp_left_swing = DMP(y_des_left_swing, left_swing_time, ay=ay, by=by, n_bfs=150, dt=0.001, isz=True)
    _, dmp_time_left_swing = dmp_left_swing.imitate_path()
    dmp_right_swing = DMP(y_des_right_swing, right_swing_time, ay=ay, by=by, n_bfs=150, dt=0.001, isz=True)
    _, dmp_time_right_swing = dmp_right_swing.imitate_path()
    dmp_left_swing.save_para(f"./dmp_temp/dmp_left_swing")
    dmp_right_swing.save_para(f"./dmp_temp/dmp_right_swing")
    dmp_time_right_swing += dmp_time_left_swing[-1] + 0.001

    # Verify dmp fitting.
    goal, tau = 1.0, 1.0
    track_res_left_swing, track_res_vel_left_swing, _, time_left_swing = dmp_left_swing.rollout(goal=goal, tau=tau)
    track_res_right_swing, track_res_vel_right_swing, _, time_right_swing = dmp_right_swing.rollout(goal=goal, tau=tau)

    fig, axs = plt.subplots(2,2,sharey="row")
    axs[0,0].set_title("DMP Fitting Left Swing - Hip")
    axs[0,0].set_xlabel("Time (sec)")
    axs[0,0].set_ylabel("Degree")
    axs[0,0].plot(left_swing_time, rhip_lsw, lw=5, label="right hip joint ideal")
    axs[0,0].plot(left_swing_time, lhip_lsw, lw=5, label="left hip joint ideal")
    axs[0,0].plot(time_left_swing, track_res_left_swing[:, 0], lw=1, label="right hip joint dmp")
    axs[0,0].plot(time_left_swing, track_res_left_swing[:, 1], lw=1, label="left hip joint dmp")
    axs[1,0].set_title("DMP Fitting Left Swing - Knee")
    axs[1,0].set_xlabel("Time (sec)")
    axs[1,0].set_ylabel("Degree")
    axs[1,0].plot(left_swing_time, rknee_lsw, lw=5, label="right knee joint ideal")
    axs[1,0].plot(left_swing_time, lknee_lsw, lw=5, label="left knee joint ideal")
    axs[1,0].plot(time_left_swing, track_res_left_swing[:, 2], lw=1, label="right knee joint dmp")
    axs[1,0].plot(time_left_swing, track_res_left_swing[:, 3], lw=1, label="left knee joint dmp")
    axs[0,1].set_title("DMP Fitting Right Swing - Hip")
    axs[0,1].set_xlabel("Time (sec)")
    axs[0,1].set_ylabel("Degree")
    axs[0,1].plot(right_swing_time, rhip_rsw, lw=5, label="right hip joint ideal")
    axs[0,1].plot(right_swing_time, lhip_rsw, lw=5, label="left hip joint ideal")
    axs[0,1].plot(time_right_swing, track_res_right_swing[:, 0], lw=1, label="right hip joint dmp")
    axs[0,1].plot(time_right_swing, track_res_right_swing[:, 1], lw=1, label="left hip joint dmp")
    axs[1,1].set_title("DMP Fitting Right Swing - Knee")
    axs[1,1].set_xlabel("Time (sec)")
    axs[1,1].set_ylabel("Degree")
    axs[1,1].plot(right_swing_time, rknee_rsw, lw=5, label="right knee joint ideal")
    axs[1,1].plot(right_swing_time, lknee_rsw, lw=5, label="left knee joint ideal")
    axs[1,1].plot(time_right_swing, track_res_right_swing[:, 2], lw=1, label="right knee joint dmp")
    axs[1,1].plot(time_right_swing, track_res_right_swing[:, 3], lw=1, label="left knee joint dmp")
    for ax in axs:
        for a in ax:
            a.legend()
    plt.show()

    # # Save the trajectories.
    # np.savetxt(f"dmp_temp/rhip_lsw.txt", track_res_left_swing[:, 0])
    # np.savetxt(f"dmp_temp/lhip_lsw.txt", track_res_left_swing[:, 1])
    # np.savetxt(f"dmp_temp/rknee_lsw.txt", track_res_left_swing[:, 2])
    # np.savetxt(f"dmp_temp/lknee_lsw.txt", track_res_left_swing[:, 3])

    # np.savetxt(f"dmp_temp/rhip_rsw.txt", track_res_right_swing[:, 0])
    # np.savetxt(f"dmp_temp/lhip_rsw.txt", track_res_right_swing[:, 1])
    # np.savetxt(f"dmp_temp/rknee_rsw.txt", track_res_right_swing[:, 2])
    # np.savetxt(f"dmp_temp/lknee_rsw.txt", track_res_right_swing[:, 3])

    # Save the trajectories
    with open("./dmp_temp/left_swing_angle_trajectory.txt", mode="w", newline="") as data_file:
        writer = csv.writer(data_file, delimiter=",")
        data_file.write(f"#size: {track_res_left_swing.shape[0]} 4\n")
        data_file.write(f"#right_hip, left_hip, right_knee, left_knee\n")
        for i in range(track_res_left_swing.shape[0]):
            writer.writerow(track_res_left_swing[i, :])

    with open("./dmp_temp/right_swing_angle_trajectory.txt", mode="w", newline="") as data_file:
        writer = csv.writer(data_file, delimiter=",")
        data_file.write(f"#size: {track_res_right_swing.shape[0]} 4\n")
        data_file.write(f"#right_hip, left_hip, right_knee, left_knee\n")
        for i in range(track_res_right_swing.shape[0]):
            writer.writerow(track_res_right_swing[i, :])
else:
    data_file_path = "./machine_learning/gait/data/SN_all_subj_obstacle_first.txt"
    st_hip, st_knee, sw_hip, sw_knee, times = ReadDataFromText(data_file_path)
    #Plot hip and knee joint angle
    f, axs = plt.subplots(2,2)
    axs[0][0].plot(times, st_hip, label='hip angle in stance')
    axs[0][0].plot(times, st_knee, label='knee angle in stance')
    axs[0][1].plot(times, sw_hip, label='hip angle in swing')
    axs[0][1].plot(times, sw_knee, label='knee angle in swing')
    for ax in axs:
        for a in ax:
            a.legend()
    
    # DMP fitting
    y_des = np.array([st_hip, sw_hip, st_knee, sw_knee])
    y_des_time = times
    
    ay = np.ones(4) * 50
    by = ay / 4
    dmp_gait = DMP(y_des, y_des_time, ay=ay, by=by, n_bfs=150, dt=0.001, isz=True)
    _, dmp_time_left_swing = dmp_gait.imitate_path()
    
    # Saving parameter files
    dmp_gait.save_para(f"./dmp_temp/dmp_obstacle_first")
    
    
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
    
    # Generate target trajectory using DMP step by step
    for i in range(num_steps):
        gait_phase = float(i) / num_steps
        
        y, dy, ddy = dmp_gait.step_real(gait_phase, y, dy, scale=new_scale, goal_offset=goal_offset, tau=1.0, extra_force=0.0)
        track[:, i] = deepcopy(y)
    
    dmp_st_hip = track[0,:]
    dmp_sw_hip = track[1,:]
    dmp_st_knee = track[2,:]
    dmp_sw_knee = track[3,:]
    # Plot original and generated trajectories
    f, axs = plt.subplots(2,2)
    axs[0][0].plot(times, st_hip, label='st_hip')
    axs[0][0].plot(dmp_time, dmp_st_hip, label='dmp st_hip')
    axs[1][0].plot(times, st_knee, label='st_knee')
    axs[1][0].plot(dmp_time, dmp_st_knee, label='dmp st_knee')
    
    axs[0][1].plot(times, sw_hip, label='sw_hip')
    axs[0][1].plot(dmp_time, dmp_sw_hip, label='dmp sw_hip')
    axs[1][1].plot(times, sw_knee, label='sw_knee')
    axs[1][1].plot(dmp_time, dmp_sw_knee, label='dmp sw_knee')
    
    for ax in axs:
        for a in ax:
            a.legend()
    plt.show()
    
    
#Use end effector
    # y_des_left_swing = np.array([right_ankle_left_swing[0], right_ankle_left_swing[2], left_ankle_left_swing[0], left_ankle_left_swing[2]])
    # y_des_right_swing = np.array([right_ankle_right_swing[0], right_ankle_right_swing[2], left_ankle_right_swing[0], left_ankle_right_swing[2]])
    # print(y_des_right_swing.shape)
    # sys.exit()
    # ay = np.ones(4) * 100
    # by = ay / 4
    # dmp_left_swing = DMP(y_des_left_swing, left_swing_time, ay=ay, by=by, n_bfs=350, dt=0.001, isz=True)
    # _, dmp_time_left_swing = dmp_left_swing.imitate_path()
    # dmp_right_swing = DMP(y_des_right_swing, right_swing_time, ay=ay, by=by, n_bfs=350, dt=0.001, isz=True)
    # _, dmp_time_right_swing = dmp_right_swing.imitate_path()
    # dmp_left_swing.save_para(f"./dmp_temp/dmp_left_swing")
    # dmp_right_swing.save_para(f"./dmp_temp/dmp_right_swing")
    # dmp_time_right_swing += dmp_time_left_swing[-1] + 0.001

    # # Verify dmp fitting.
    # goal, tau = 1.0, 1.0
    # track_res_left_swing, track_res_vel_left_swing, _, time_left_swing = dmp_left_swing.rollout(goal=goal, tau=tau)
    # track_res_right_swing, track_res_vel_right_swing, _, time_right_swing = dmp_right_swing.rollout(goal=goal, tau=tau)
    
    # fig, axs = plt.subplots(1,2,sharey="row")
    # axs[0].set_title("DMP Fitting Left Swing - Ankle")
    # axs[0].set_xlabel("X")
    # axs[0].set_ylabel("Z")
    # axs[0].plot(right_ankle_left_swing[0], right_ankle_left_swing[2], ls='--', label="right ankle ideal")
    # axs[0].plot(left_ankle_left_swing[0], left_ankle_left_swing[2], ls='--', label="left ankle ideal")
    # axs[0].plot(track_res_left_swing[:, 0], track_res_left_swing[:, 1], label="right ankle dmp")
    # axs[0].plot(track_res_left_swing[:, 2], track_res_left_swing[:, 3],  label="left ankle dmp")
    # axs[1].set_title("DMP Fitting Right Swing - Ankle")
    # axs[1].set_xlabel("X")
    # axs[1].set_ylabel("Z")
    # axs[1].plot(right_ankle_right_swing[0], right_ankle_right_swing[2], ls='--', label="right ankle ideal")
    # axs[1].plot(left_ankle_right_swing[0], left_ankle_right_swing[2], ls='--', label="left ankle ideal")
    # axs[1].plot(track_res_right_swing[:, 0], track_res_right_swing[:, 1], label="right ankle dmp")
    # axs[1].plot(track_res_right_swing[:, 2], track_res_right_swing[:, 3],  label="left ankle dmp")
    # for ax in axs:
    #     ax.legend()
    # plt.show()
    
    # # Save the trajectories
    # with open("./dmp_temp/left_swing_trajectory.txt", mode="w", newline="") as data_file:
    #     writer = csv.writer(data_file, delimiter=",")
    #     data_file.write(f"#size: {track_res_left_swing.shape[0]} 4\n")
    #     data_file.write(f"#right_ankle_x, right_ankle_z, left_ankle_x, left_ankle_z\n")
    #     for i in range(track_res_left_swing.shape[0]):
    #         writer.writerow(track_res_left_swing[i, :])

    # with open("./dmp_temp/right_swing_trajectory.txt", mode="w", newline="") as data_file:
    #     writer = csv.writer(data_file, delimiter=",")
    #     data_file.write(f"#size: {track_res_right_swing.shape[0]} 4\n")
    #     data_file.write(f"#right_ankle_x, right_ankle_z, left_ankle_x, left_ankle_z\n")
    #     for i in range(track_res_right_swing.shape[0]):
    #         writer.writerow(track_res_right_swing[i, :])

