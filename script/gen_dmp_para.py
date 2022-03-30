from pathlib import Path

from copy import deepcopy
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate

from sim import EXO_SIM
from dmp import DMP
from utils import bodyM, ReadDataFromCSV, ReadDataFromText, CorrectHipForStepLength, r_2_d, slice_gait_phase
import csv
import sys


is_data_from_mocop = False
is_correct_hip = False
is_save_file = False
path = "D:/MyFolder/code/gait_ws/gait_LLE/data"
load_file_name = "SN_all_subj_obstacle_first" #"DMP_left_swing" #"SN002_0028_10m_01.csv" #  #
saved_file_name = "left_swing_1"

# load data
if is_data_from_mocop is True:
    data_file_path = path + "/mocap/" + load_file_name
    st_hip, sw_hip, st_knee, sw_knee, times = ReadDataFromCSV(data_file_path)
else:
    data_file_path = path + "/joint_angle/" + load_file_name + ".txt"
    st_hip, sw_hip, st_knee, sw_knee, times = ReadDataFromText(data_file_path)

# DMP creating and training
y_des = np.array([st_hip, sw_hip, st_knee, sw_knee])
y_des_time = times

ay = np.ones(4) * 50
by = ay / 4
dmp_gait = DMP(y_des, y_des_time, n_bfs=200, dt=0.001, isz=True)
_, _, = dmp_gait.imitate_path()

# Saving DMP parameters
if is_save_file is True:
    dmp_gait.save_para(path + "/dmp_para/" + saved_file_name)

# DMP fitting results and plotting
goal, tau = 1.0, 1.0
y_track, dy_track, ddy_track, dmp_time = dmp_gait.rollout(goal=goal, tau=tau)

plt.figure(1)
plt.subplot(121)
plt.title("DMP Fitting Stance leg")
plt.xlabel("Time (sec)")
plt.ylabel("Degree")
plt.plot(y_des_time, st_hip, lw=3, label="ideal stance hip")
plt.plot(y_des_time, st_knee, lw=3, label="ideal stance knee")
plt.plot(dmp_time, y_track[:, 0], lw=1, label="dmp stance hip")
plt.plot(dmp_time, y_track[:, 2], lw=1, label="dmp stance knee")
plt.legend()
plt.subplot(122)
plt.title("DMP Fitting Swing leg")
plt.xlabel("Time (sec)")
plt.ylabel("Degree")
plt.plot(y_des_time, sw_hip, lw=3, label="ideal swing hip")
plt.plot(y_des_time, sw_knee, lw=3, label="ideal swing knee")
plt.plot(dmp_time, y_track[:, 1], lw=1, label="dmp swing hip")
plt.plot(dmp_time, y_track[:, 3], lw=1, label="dmp swing knee")
plt.legend()

########  Test DMP with different y0, scale, goal_offset   #############
traj_num = y_des.shape[0]
num_steps = 500
tau = (dmp_gait.timesteps+1) / num_steps

track = np.zeros((traj_num, num_steps))
track_time = np.arange(num_steps) * dmp_gait.dt * tau

# the goal_offset, scale, and initial position for the generated trajectory
goal_offset = np.array([0,0,0,0])
new_scale = np.ones(traj_num)
y = y_des[:,0] + [0,0,0,0]
dy = np.zeros(traj_num)
ddy = np.zeros(traj_num)


# since two ankle joints are not on the same plane, it needs to correct hip joint angle
if is_correct_hip is True:
    goal_angle = dmp_gait.goal
    st_hip_goal, sw_hip_goal = CorrectHipForStepLength(goal_angle,0.44,0.565)
    goal_offset[0] = st_hip_goal-goal_angle[0]
    goal_offset[1] = sw_hip_goal-goal_angle[1]
    y = y_des[:,0] + [sw_hip_goal-goal_angle[1],st_hip_goal-goal_angle[0],0,0]


# Generate target trajectory using DMP step by step
for i in range(num_steps):
    gait_phase = float(i) / num_steps
    y, dy, ddy = dmp_gait.step_real(gait_phase, y, dy, scale=new_scale, goal_offset=goal_offset, tau=tau)
    track[:, i] = deepcopy(y)

dmp_st_hip = track[0,:]
dmp_sw_hip = track[1,:]
dmp_st_knee = track[2,:]
dmp_sw_knee = track[3,:]

# Plot original and generated trajectories
f, axs = plt.subplots(2,2)
axs[0][0].plot(y_des_time, st_hip, label='st_hip')
axs[0][0].plot(track_time, dmp_st_hip, label='dmp st_hip')
axs[1][0].plot(y_des_time, st_knee, label='st_knee')
axs[1][0].plot(track_time, dmp_st_knee, label='dmp st_knee')

axs[0][1].plot(y_des_time, sw_hip, label='sw_hip')
axs[0][1].plot(track_time, dmp_sw_hip, label='dmp sw_hip')
axs[1][1].plot(y_des_time, sw_knee, label='sw_knee')
axs[1][1].plot(track_time, dmp_sw_knee, label='dmp sw_knee')

for ax in axs:
    for a in ax:
        a.legend()
plt.show()

# Save the trajectories in text files
if is_save_file is True:
    with open(path + "/joint_angle/dmp_" + saved_file_name + ".txt", mode="w", newline="") as data_file:
        writer = csv.writer(data_file, delimiter=",")
        data_file.write(f"#size: {track.shape[1]} 4\n")
        data_file.write(f"#stance_hip, swing_hip, stance_knee, swing_knee\n")
        for i in range(track.shape[1]):
            writer.writerow(track[:,i])

plt.figure()
exo = EXO_SIM()
exo.update_stace_leg(True)
exo.plot_one_step(dmp_st_hip,dmp_st_knee,dmp_sw_hip,dmp_sw_knee,"levelground")
plt.show()