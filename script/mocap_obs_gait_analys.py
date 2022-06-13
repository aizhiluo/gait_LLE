"""
This aiming is to read all subjects' gait data in different obstacles crossing from data files.
The data includes joint angles and ankle position. The raw data and mean value over subjects are plotted.
Then, the raw data is saved into .npy file.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import csv

load_data_flag = True # if true, directly load data from .npy file, otherwise read data from raw data files. 
is_saving_template = False

subj_tight_shank = [[0.410,0.374],[0.507,0.415],[0.518,0.424],[0.497,0.459],[0.467,0.459]]

path = 'C:/Users/lincong.luo/Desktop/Mocap_data/5subj/'
file = ['static_ob_5x10_AnkleTrajectoryAndJointAngles','static_ob_10x10_AnkleTrajectoryAndJointAngles','static_ob_10x20_AnkleTrajectoryAndJointAngles','static_ob_10x30_AnkleTrajectoryAndJointAngles','static_ob_20x10_AnkleTrajectoryAndJointAngles']
sheets_name = ['All_Subjs-ankle_pos_aligned','All_Subjs-joint_angle_aligned']
columns = [['Stance0X','Stance0Y','Stance0Z','Swing0X','Swing0Y','Swing0Z'],['Stance1X','Stance1Y','Stance1Z','Swing1X','Swing1Y','Swing1Z']]
column_ang = [['Stance0Hip','Swing0Hip','Stance0Knee','Swing0Knee'],['Stance1Hip','Swing1Hip','Stance1Knee','Swing1Knee']]


# swing ankle position relative to the stance ankle
all_subj_relative_ankle_px = np.zeros((5,5,500)) # obs * subj * gait points num
all_subj_relative_ankle_pz = np.zeros((5,5,500))

# hip and knee of swing and stance legs
all_subj_st_hip = np.zeros((5,5,500))
all_subj_sw_hip = np.zeros((5,5,500))
all_subj_st_knee = np.zeros((5,5,500))
all_subj_sw_knee = np.zeros((5,5,500))

# ankle position relative to hip joint
all_subj_st_px = np.zeros((5,5,500))
all_subj_st_pz = np.zeros((5,5,500))
all_subj_sw_px = np.zeros((5,5,500))
all_subj_sw_pz = np.zeros((5,5,500))


if load_data_flag is True:
# load data from .npy files
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
else:
    # read data from .xlsx files
    for i in range(5):
        df = pd.read_excel(path+file[i]+'.xlsx', sheet_name=sheets_name[0])
        all_ankle_pos = np.array(df[['Stance0X','Stance0Z','Swing0X','Swing0Z']].values.tolist()).T
        
        df_ang = pd.read_excel(path+file[i]+'.xlsx', sheet_name=sheets_name[1])
        all_joint_ang = np.array(df_ang[['Stance0Hip','Swing0Hip','Stance0Knee','Swing0Knee']].values.tolist()).T
        
        # 5 subjects, every subject conducts crossing step 4 times
        for j in range(5):
            sum_pos = np.zeros((4,500)) # [px,pz]
            sum_ang = np.zeros((4,500)) # st_hip, sw_hip, st_knee, sw_knee
            
            print(i,j)
            # mean in a gait cycle
            for k in range(4):
                st = j*2000 + k*500
                en = j*2000 + k*500 + 500
                if i==2 and j==4 and k==3: # skiping the cycle which lacks of gait data
                    break
                sum_pos += all_ankle_pos[:,st:en]
                sum_ang += all_joint_ang[:,st:en]
            
            # the file lacking one cycle data
            if i==2 and j==4:
                sum_pos = sum_pos / 3.0
                sum_ang = sum_ang / 3.0
            else:
                sum_pos = sum_pos / 4.0
                sum_ang = sum_ang / 4.0
            
            all_subj_st_hip[i,j,:] = sum_ang[0,:]
            all_subj_sw_hip[i,j,:] = sum_ang[1,:]
            all_subj_st_knee[i,j,:] = sum_ang[2,:]
            all_subj_sw_knee[i,j,:] = sum_ang[3,:]
            all_subj_st_px[i,j,:]  = sum_pos[0,:]
            all_subj_st_pz[i,j,:]  = sum_pos[1,:]
            all_subj_sw_px[i,j,:]  = sum_pos[2,:]
            all_subj_sw_pz[i,j,:]  = sum_pos[3,:]
            
            all_subj_relative_ankle_px = all_subj_sw_px - all_subj_st_px
            all_subj_relative_ankle_pz = all_subj_sw_pz - all_subj_st_pz

    # save data
    with open('obst_first_step_all_subj_data.npy','wb') as f:
        np.save(f,all_subj_st_hip)
        np.save(f,all_subj_sw_hip)
        np.save(f,all_subj_st_knee)
        np.save(f,all_subj_sw_knee)
        np.save(f,all_subj_st_px)
        np.save(f,all_subj_st_pz)
        np.save(f,all_subj_sw_px)
        np.save(f,all_subj_sw_pz)
    
# mean value among all subjects
mean_subj_st_hip = all_subj_st_hip.mean(axis=1)
mean_subj_sw_hip = all_subj_sw_hip.mean(axis=1)
mean_subj_st_knee = all_subj_st_knee.mean(axis=1)
mean_subj_sw_knee = all_subj_sw_knee.mean(axis=1)

mean_subj_st_hip_px = -all_subj_st_px.mean(axis=1) # the hip position relative to the stance ankle
mean_subj_st_hip_pz = -all_subj_st_pz.mean(axis=1)
mean_subj_relative_px = all_subj_relative_ankle_px.mean(axis=1) # the swing ankle position relative to the stance ankle
mean_subj_relative_pz = all_subj_relative_ankle_pz.mean(axis=1)


""" plot mean data over subjects in different obstacles """
# # plot average trajectories
# index = [[0,1,4],[1,2,3]]
# title_size = 20
# label_size = 18
# f, axs = plt.subplots(2,2)
# f.suptitle('average stance joint angles',fontsize = title_size*1.2)
# axs[0][0].set_title('different obstacle height',fontsize = title_size)
# axs[0][1].set_title('different obstacle width',fontsize = title_size)
# axs[0][0].set_ylabel('st_hip angle',fontsize = label_size)
# axs[0][1].set_ylabel('st_hip angle',fontsize = label_size)
# axs[1][0].set_ylabel('st_knee angle',fontsize = label_size)
# axs[1][1].set_ylabel('st_knee angle',fontsize = label_size)

# axs[0][0].plot(mean_subj_st_hip[0,:],label='5x10')
# axs[0][0].plot(mean_subj_st_hip[1,:],label='10x10')
# axs[0][0].plot(mean_subj_st_hip[4,:],label='20x10')
# axs[1][0].plot(mean_subj_st_knee[0,:],label='5x10')
# axs[1][0].plot(mean_subj_st_knee[1,:],label='10x10')
# axs[1][0].plot(mean_subj_st_knee[4,:],label='20x10')

# axs[0][1].plot(mean_subj_st_hip[1,:],label='10x10')
# axs[0][1].plot(mean_subj_st_hip[2,:],label='10x20')
# axs[0][1].plot(mean_subj_st_hip[3,:],label='10x30')
# axs[1][1].plot(mean_subj_st_knee[1,:],label='10x10')
# axs[1][1].plot(mean_subj_st_knee[2,:],label='10x20')
# axs[1][1].plot(mean_subj_st_knee[3,:],label='10x30')

# for ax in axs:
#         for a in ax:
#             a.legend()

# f1, axs1 = plt.subplots(2,2)
# f1.suptitle('average swing position',fontsize = title_size*1.2)
# axs1[0][0].set_title('different obstacle height',fontsize = title_size)
# axs1[0][1].set_title('different obstacle width',fontsize = title_size)
# axs1[0][0].set_ylabel('sw_ankle X',fontsize = label_size)
# axs1[0][1].set_ylabel('sw_ankle X',fontsize = label_size)
# axs1[1][0].set_ylabel('sw_ankle Z',fontsize = label_size)
# axs1[1][1].set_ylabel('sw_ankle Z',fontsize = label_size)

# axs1[0][0].plot(mean_subj_relative_px[0,:],label='5x10')
# axs1[0][0].plot(mean_subj_relative_px[1,:],label='10x10')
# axs1[0][0].plot(mean_subj_relative_px[4,:],label='20x10')
# axs1[1][0].plot(mean_subj_relative_pz[0,:],label='5x10')
# axs1[1][0].plot(mean_subj_relative_pz[1,:],label='10x10')
# axs1[1][0].plot(mean_subj_relative_pz[4,:],label='20x10')

# axs1[0][1].plot(mean_subj_relative_px[1,:],label='10x10')
# axs1[0][1].plot(mean_subj_relative_px[2,:],label='10x20')
# axs1[0][1].plot(mean_subj_relative_px[3,:],label='10x30')
# axs1[1][1].plot(mean_subj_relative_pz[1,:],label='10x10')
# axs1[1][1].plot(mean_subj_relative_pz[2,:],label='10x20')
# axs1[1][1].plot(mean_subj_relative_pz[3,:],label='10x30')

# for ax in axs1:
#         for a in ax:
#             a.legend()

# f2, axs2 = plt.subplots(2,2)
# f2.suptitle('average stance hip position',fontsize = title_size*1.2)
# axs2[0][0].set_title('different obstacle height',fontsize = title_size)
# axs2[0][1].set_title('different obstacle width',fontsize = title_size)
# axs2[0][0].set_ylabel('sw_ankle X',fontsize = label_size)
# axs2[0][1].set_ylabel('sw_ankle X',fontsize = label_size)
# axs2[1][0].set_ylabel('sw_ankle Z',fontsize = label_size)
# axs2[1][1].set_ylabel('sw_ankle Z',fontsize = label_size)

# axs2[0][0].plot(mean_subj_st_hip_px[0,:],label='5x10')
# axs2[0][0].plot(mean_subj_st_hip_px[1,:],label='10x10')
# axs2[0][0].plot(mean_subj_st_hip_px[4,:],label='20x10')
# axs2[1][0].plot(mean_subj_st_hip_pz[0,:],label='5x10')
# axs2[1][0].plot(mean_subj_st_hip_pz[1,:],label='10x10')
# axs2[1][0].plot(mean_subj_st_hip_pz[4,:],label='20x10')

# axs2[0][1].plot(mean_subj_st_hip_px[1,:],label='10x10')
# axs2[0][1].plot(mean_subj_st_hip_px[2,:],label='10x20')
# axs2[0][1].plot(mean_subj_st_hip_px[3,:],label='10x30')
# axs2[1][1].plot(mean_subj_st_hip_pz[1,:],label='10x10')
# axs2[1][1].plot(mean_subj_st_hip_pz[2,:],label='10x20')
# axs2[1][1].plot(mean_subj_st_hip_pz[3,:],label='10x30')

# for ax in axs2:
#         for a in ax:
#             a.legend()


# obs_num = 0
# obs_name = ['5x10','10x10','10x20','10x30','20x10']
# f3, axs3 = plt.subplots(2,2)
# f3.suptitle('template trajectory',fontsize = title_size*1.2)
# axs3[0][0].set_title('stance knee angle',fontsize = title_size)
# axs3[1][0].set_title('hip position in px',fontsize = title_size)
# axs3[0][1].set_title('swing ankle px',fontsize = title_size)
# axs3[1][1].set_title('swing ankle pz',fontsize = title_size)

# for i in range(5):
#     axs3[0][0].plot(mean_subj_st_knee[i,:],label=obs_name[i])
#     axs3[1][0].plot(mean_subj_st_hip_px[i,:],label=obs_name[i])
#     axs3[0][1].plot(mean_subj_relative_px[i,:],label=obs_name[i])
#     axs3[1][1].plot(mean_subj_relative_pz[i,:],label=obs_name[i])

# axs3[0][0].plot(mean_subj_st_knee.mean(axis=0),'--',label='mean')

# for ax in axs3:
#         for a in ax:
#             a.legend()



""" plot all subject and obstacles """
# figure1 plot swing ankle position relative to stance ankle
index = [0,1,4,1,2,3]
f, axs = plt.subplots(2,3)
f.suptitle('swing ankle position relative to hip')
axs[0][0].set_title('5x10')
axs[0][1].set_title('10x10')
axs[0][2].set_title('20x10')
axs[1][0].set_title('10x10')
axs[1][1].set_title('10x20')
axs[1][2].set_title('10x30')
for i in range(6):
    row = i // 3
    col = i - row * 3
    k = index[i]
    axs[row][col].set_xlim([-0.05, 0.75])
    axs[row][col].set_ylim([-0.05, 0.60])
    axs[row][col].set_xlabel('px/m')
    axs[row][col].set_ylabel('pz/m')
    axs[row][col].grid(True)
    for j in range(5):
        c = 'C'+str(j)
        axs[row][col].plot(all_subj_relative_ankle_px[k,j,:], all_subj_relative_ankle_pz[k,j,:], label='sub'+str(j+1))
for ax in axs:
        for a in ax:
            a.legend()
            
# figure2 plot stance join angles with different obstacles
index = [0,1,4,1,2,3]
f1, axs1 = plt.subplots(2,3)
f1.suptitle('hip position relative to stance ankle')
axs1[0][0].set_title('5x10')
axs1[0][1].set_title('10x10')
axs1[0][2].set_title('20x10')
axs1[1][0].set_title('10x10')
axs1[1][1].set_title('10x20')
axs1[1][2].set_title('10x30')
for i in range(6):
    row = i // 3
    col = i - row * 3
    k = index[i]
    # axs1[row][col].set_ylim([-25.0, 35.0])
    axs1[row][col].set_xlabel('PX')
    axs1[row][col].set_ylabel('PZ')
    axs1[row][col].grid(False)
    for j in range(5):
        c = 'C'+str(j)
        axs1[row][col].plot(-all_subj_st_px[k,j,:],-all_subj_st_pz[k,j,:],c,label='sub'+str(j+1))      
for ax in axs1:
        for a in ax:
            a.legend()
            a.grid(True)
            
# figure3 plot stance ankle px pz position
index = [0,1,4,1,2,3]
f2, axs2 = plt.subplots(2,3)
f2.suptitle('swing ankle position relative to stance ankle')
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
    axs2[row][col].set_xlabel('px/m')
    axs2[row][col].set_ylabel('pz/m')
    axs2[row][col].grid(True)
    for j in range(5):
        c = 'C'+str(j)
        axs2[row][col].plot(all_subj_relative_ankle_px[k,j,:],c,label='sub'+str(j+1)+'-px')
        axs2[row][col].plot(all_subj_relative_ankle_pz[k,j,:],c+'--', label='sub'+str(j+1)+'-pz')
for ax in axs2:
        for a in ax:
            a.legend()
            
# plot hip joint position
index = [0,1,4,1,2,3]
f3, axs3 = plt.subplots(2,3)
f3.suptitle('hip position relative to stance ankle')
axs3[0][0].set_title('5x10')
axs3[0][1].set_title('10x10')
axs3[0][2].set_title('20x10')
axs3[1][0].set_title('10x10')
axs3[1][1].set_title('10x20')
axs3[1][2].set_title('10x30')
for i in range(6):
    row = i // 3
    col = i - row * 3
    k = index[i]
    # axs3[row][col].set_xlim([-0.05, 0.75])
    # axs3[row][col].set_ylim([-0.05, 0.60])
    axs3[row][col].set_xlabel('px/m')
    axs3[row][col].set_ylabel('pz/m')
    axs3[row][col].grid(True)
    for j in range(5):
        c = 'C'+str(j)
        axs3[row][col].plot(-all_subj_st_px[k,j,:], c, label='sub'+str(j+1)+'-px')
        axs3[row][col].plot(subj_tight_shank[j][0]+subj_tight_shank[j][1]+all_subj_st_pz[k,j,:], c+'--', label='sub'+str(j+1)+'-pz')
for ax in axs3:
        for a in ax:
            a.legend()

# plot stance hip and knee angles
index = [0,1,4,1,2,3]
f4, axs4 = plt.subplots(2,3)
f4.suptitle('stance hip and knee angles')
axs4[0][0].set_title('5x10')
axs4[0][1].set_title('10x10')
axs4[0][2].set_title('20x10')
axs4[1][0].set_title('10x10')
axs4[1][1].set_title('10x20')
axs4[1][2].set_title('10x30')
for i in range(6):
    row = i // 3
    col = i - row * 3
    k = index[i]
    axs4[row][col].set_ylim([-20, 60])
    # axs4[row][col].set_xlabel('px/m')
    axs4[row][col].set_ylabel('angle/[deg]')
    axs4[row][col].grid(True)
    for j in range(5):
        c = 'C'+str(j)
        axs4[row][col].plot(all_subj_st_hip[k,j,:],c+'--',label='hip'+str(j+1))
        axs4[row][col].plot(all_subj_st_knee[k,j,:],c,label='knee'+str(j+1))

for ax in axs4:
        for a in ax:
            a.legend()
            
plt.show()

# Save the template trajectories into text files
if is_saving_template is True:
    # select 10x10 obstacle as template
    data = np.zeros((4,mean_subj_relative_px.shape[1]))
    data[0,:] = mean_subj_st_hip[1,:]
    data[1,:] = mean_subj_st_knee[1,:]
    data[2,:] = mean_subj_relative_px[1,:]
    data[3,:] = mean_subj_relative_pz[1,:]
    with open('1step_obstacle_template.txt', mode="w", newline="") as data_file:
        writer = csv.writer(data_file, delimiter=",")
        data_file.write(f"#size: {mean_subj_relative_px.shape[1]} 4\n")
        data_file.write(f"#stance_hip, stance_knee, swing_ankle_px, swing_ankle_pz\n")
        for i in range(mean_subj_relative_px.shape[1]):
            writer.writerow(data[:,i])