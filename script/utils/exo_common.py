import math
from math import pi
import numpy as np
import sys

def JacobianMatrix(hip,knee,thigh_length,shank_length):
    
    hip = hip / 180 * np.pi
    knee = knee / 180 * np.pi
    
    J = np.zeros((2,2))
    J[0,0] = -thigh_length * np.sin(hip - np.pi/2) - shank_length * np.sin(hip - np.pi/2 - knee)
    J[0,1] = shank_length * np.sin(hip - np.pi/2 - knee)
    J[1,0] = thigh_length * np.cos(hip - np.pi/2) + shank_length * np.cos(hip - np.pi/2 - knee)
    J[1,1] = -shank_length * np.cos(hip - np.pi/2 - knee)
    
    return J
    
def Kinematics(hip,knee,thigh_length,shank_length,ankle=0.0,foot_length=0.0):
    """ Kinematics:
        Calculate the foot position in Cartesian coordinate based on joint angle
        the hip joint is defined as the origin of coordinate; hip flexion is for positive angle and knee flexion is positive angle.
        Standing straight is the initial statue (hip=0 degree, knee=0 degree)
    Args:
    
    Return:
        px: 
        pz: 
    """
    hip = hip / 180 * np.pi
    knee = knee / 180 * np.pi
    ankle = ankle / 180 * np.pi
    
    px = thigh_length * np.cos(hip - np.pi/2) + shank_length * np.cos(hip - np.pi/2 - knee) + foot_length * np.cos(hip - np.pi/2 - knee + ankle)
    pz = thigh_length * np.sin(hip - np.pi/2) + shank_length * np.sin(hip - np.pi/2 - knee) + foot_length * np.sin(hip - np.pi/2 - knee + ankle)

    return px, pz

def InverseKinematics(pts,thigh_length,shank_length):
    """ Inverse Kinematics:

    Args:
        pts (2-D array): px and pz
        thigh_length (float): 
        shank_length (float): 

    Returns:
       hip [1-D array]: hip joint angle with unit of degree
       knee [1-D array]: knee joint angle with unit of degree
    """
    if pts.ndim == 2:
        num = pts.shape[1]
        ptx = pts[0,:]
        ptz = pts[1,:]
    else:
        num = 1
        ptx = np.array([pts[0]])
        ptz = np.array([pts[1]])
    
    hip = np.zeros(num)
    knee = np.zeros(num)

    for i in range(num):
        px = ptx[i]
        pz = ptz[i]
        dist = np.sqrt(px * px + pz * pz)

        #Prevent from getting out of workspace
        if (dist > shank_length + thigh_length):
            scale = (shank_length + thigh_length) / dist * 0.999
            px = px * scale
            pz = pz * scale
            print("Out of robot workspace!")
            # sys.exit()
            

        dist = np.sqrt(px * px + pz * pz)
        belt1 = np.arccos((shank_length * shank_length - thigh_length * thigh_length - dist * dist) / (-2 * dist * thigh_length))
        belt2 = np.arccos((thigh_length * thigh_length - shank_length * shank_length - dist * dist) / (-2 * shank_length * dist))
        alpha = np.arcsin(px / dist)

        hip[i] = (alpha + belt1) * 180.0 / np.pi
        knee[i] = (belt1 + belt2) * 180.0 / np.pi
    
    if num == 1:
        return hip[0], knee[0]
    else:
        return hip, knee

def CorrectHipForStepLength(angle,thigh,shank,step_length=None):
    """correct front and back leg hip angle because the twp angle is not in the same Pz

    Args:
        angle (_type_): back hip, front hip, back knee, front knee
        thigh (_type_): _description_
        shank (_type_): _description_
        step_length: expected step length
    Returns:
        _type_: _description_
    """
            
    s3_hip = angle[0]
    s4_hip = angle[1]
    s3_knee = angle[2]
    s4_knee = angle[3]
    
    s3_px,s3_pz = Kinematics(s3_hip,s3_knee,thigh,shank)
    s3 = np.sqrt(s3_px**2 + s3_pz**2)
    
    s4_px,s4_pz = Kinematics(s4_hip,s4_knee,thigh,shank)
    s4 = np.sqrt(s4_px**2 + s4_pz**2)
    
    if step_length is None:
        step_len = np.sqrt((s3_px-s4_px)**2 + (s3_pz-s4_pz)**2)
    else:
        step_len = step_length
    
    s2 = (step_len**2 + s4**2 - s3**2) / 2 / step_len
    s1 = step_len - s2
    
    theta1 = np.arccos((shank**2 - thigh**2 - s3**2) / (-2*thigh*s3))
    theta2 = np.arccos((shank**2 - thigh**2 - s4**2) / (-2*thigh*s4))
    
    back_hip = (theta1 - np.arcsin(s1/s3)) * 180 / np.pi
    front_hip = (theta2 + np.arcsin(s2/s4)) * 180 / np.pi
    
    return back_hip, front_hip

def SlopeModelNoAnkleConstrain(angle,thigh,shank,slope,step_length=None):
    
    st_h = angle[0]
    sw_h = angle[1]
    st_k = angle[2]
    sw_k = angle[3]
    
    # if no specify step length, using the input joint configuration to calculate the required step length
    if step_length is None:
        px1,pz1 = Kinematics(st_h,st_k,thigh,shank)
        px2,pz2 = Kinematics(sw_h,sw_k,thigh,shank)
        step_length = np.sqrt((px1-px2)**2 + (pz1-pz2)**2)
    
    # the position change of swing/stance foot from levelground to ascent/descent ramp
    delta_px = step_length * (1 - np.cos(slope*np.pi/180))
    delta_pz = step_length * np.sin(slope*np.pi/180)
    
    # Firstly, assume in levelground and obtain target hip joint angles
    post_st_h,post_sw_h = CorrectHipForStepLength(angle,thigh,shank,step_length)
       
    # For ascent ramp, adjust swing foot location, and stance foot location for descent ramp
    if slope > 0: 
        px,pz = Kinematics(post_sw_h,sw_k,thigh,shank)
        px = px - delta_px
        pz = pz + delta_pz
        post_sw_h,post_sw_k = InverseKinematics(np.array([px,pz]),thigh,shank)
        post_st_k = st_k
        
    else:
        # px,pz = Kinematics(post_st_h,st_k,thigh,shank)
        # px = px + delta_px
        # pz = pz - delta_pz
        # post_st_h,post_st_k = InverseKinematics(np.array([px,pz]),thigh,shank)
        # post_sw_k = sw_k
        
        # rotate whole body
        post_st_h = post_st_h + slope
        post_sw_h = post_sw_h + slope
        post_st_k = st_k
        post_sw_k = sw_k
        
    return np.array([post_st_h,post_sw_h,post_st_k,post_sw_k])

def SlopeModelWithAnkleConstrain(angle,thigh,shank,slope,step_length=None):
    st_h = angle[0]
    sw_h = angle[1]
    st_k = angle[2]
    sw_k = angle[3]
    
    # if no specify step length, using the input joint configuration to calculate the required step length
    if step_length is None:
        px1,pz1 = Kinematics(st_h,st_k,thigh,shank)
        px2,pz2 = Kinematics(sw_h,sw_k,thigh,shank)
        step_length = np.sqrt((px1-px2)**2 + (pz1-pz2)**2)
    
    # Firstly, assume in levelground and obtain target hip joint angles
    tmp_st_h,tmp_sw_h = CorrectHipForStepLength(angle,thigh,shank,step_length)
    
    # For ascent ramp:
        # first, increasing stance hip and reducing stance knee keep the angle between shank and surfance (ankle angle) is same to the one on levelground
        # second, adjusting swing hip and knee satisfy the requirement of step length
    # For descent ramp:
        # first, reducing swing hip and increasing swing knee
        # second, adjust stance leg location
    if slope > 0:
        alpha = 0.50
        tmp_st_h = tmp_st_h + slope * alpha
        tmp_st_k = st_k - slope * (1-alpha)
        
        # hip position relative to stance foot 
        com_px,com_pz = Kinematics(tmp_st_h,tmp_st_k,thigh,shank)
        com_px = -com_px
        com_pz = -com_pz
        
        # the expected swing foot location relative to stance foot (0,0)
        swing_location_px = step_length * np.cos(slope*np.pi/180)
        swing_location_pz = step_length * np.sin(slope*np.pi/180)
        
        px = swing_location_px - com_px
        pz = swing_location_pz - com_pz
        post_sw_h,post_sw_k = InverseKinematics(np.array([px,pz]),thigh,shank)
        post_st_h = tmp_st_h
        post_st_k = tmp_st_k
    else:
        alpha = 0.50
        tmp_sw_h = tmp_sw_h + slope * alpha
        tmp_sw_k = sw_k - slope * (1-alpha)
        # hip position relative to swing foot
        com_px,com_pz = Kinematics(tmp_sw_h,tmp_sw_k,thigh,shank)
        com_px = -com_px
        com_pz = -com_pz
        # the expected stance foot location relative to swing foot (0,0)
        stance_location_px = - step_length * np.cos(slope*np.pi/180)
        stance_location_pz = - step_length * np.sin(slope*np.pi/180)
        px = stance_location_px - com_px
        pz = stance_location_pz - com_pz
        post_st_h,post_st_k = InverseKinematics(np.array([px,pz]),thigh,shank)
        post_sw_h = tmp_sw_h
        post_sw_k = tmp_sw_k
        
    return np.array([post_st_h,post_sw_h,post_st_k,post_sw_k])