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
    hip = np.zeros(pts.shape[1])
    knee = np.zeros(pts.shape[1])

    for i in range(pts.shape[1]):
        px = pts[0,i]
        pz = pts[1,i]
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

    return hip, knee

def CorrectHipForStepLength(angle,thigh,shank):
    """correct front and back leg hip angle because the twp angle is not in the same Pz

    Args:
        angle (_type_): back hip, front hip, back knee, front knee
        thigh (_type_): _description_
        shank (_type_): _description_

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
    
    step_len = np.sqrt((s3_px-s4_px)**2 + (s3_pz-s4_pz)**2)
    s2 = (step_len**2 + s4**2 - s3**2) / 2 / step_len
    s1 = step_len - s2
    
    theta1 = np.arccos((shank**2 - thigh**2 - s3**2) / (-2*thigh*s3))
    theta2 = np.arccos((shank**2 - thigh**2 - s4**2) / (-2*thigh*s4))
    
    back_hip = (theta1 - np.arcsin(s1/s3)) * 180 / np.pi
    front_hip = (theta2 + np.arcsin(s2/s4)) * 180 / np.pi
    
    return back_hip, front_hip