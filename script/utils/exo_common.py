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