import math
from math import pi

import numpy as np


def cart_joint(xp, yp, l1, l2, if_debug=False):
    """ Convert from cartesian space to joint space.

	From cartesian coordinates to joint space. For equations, plz refer to notes.
	yp should be biased w.r.t ground, i.e origin at hip.

	Args:
		xp, float or np.ndarray, x coordinate of ankle.
		yp, float or np.ndarray, y coordinate of ankle.
		l1, float length of thign.
		l2, float length of shank

	Return:
		angle of hip, angle of knee. In np.ndarray or float.

	Raise:
		Will raise whenever wrong value encounted. (acos(val) where val > 1)

	"""
    if isinstance(xp, float):
        atan2 = math.atan2
        acos = math.acos
        power = math.pow
        sqrt = math.sqrt
        cos = math.cos
        sin = math.sin
    else:
        atan2 = np.arctan2
        acos = np.arccos
        power = np.power
        sqrt = np.sqrt
        cos = np.cos
        sin = np.sin

    thetap = atan2(yp, xp)
    A = power(xp, 2) + power(yp, 2)
    B1 = l1 ** 2 - l2 ** 2
    B2 = l2 ** 2 + l1 ** 2
    C = sqrt(A)
    thetah = thetap + acos((B1 + A) / (2 * l1 * C))
    thetak = -np.pi + acos((B2 - A) / (2 * l1 * l2))

    if np.isnan(thetah).any() or np.isnan(thetak).any():
        raise ValueError("NAN occurs!")

    if if_debug:
        print("Angle of hip")
        print(r_2_d(thetah))
        print("Angle of knee")
        print(d_2_r(thetak))
        xp_temp = l1 * cos(thetah) + l2 * cos(thetah + thetak)
        yp_temp = l1 * sin(thetah) + l2 * sin(thetah + thetak)
        print("difference between x")
        print(xp_temp - xp)
        print("difference between y")
        print(yp_temp - yp)

    return thetah, thetak


def joint_cart(theta1, theta2, l1, l2):
    """ Convert from joint space to cartesian space.

	From joint coordinates to cartesian space. For equations, plz refer to notes.

	Args:
		theta1, float or np.ndarray, hip angle, 0 for parallel to ground.
		theta1, float or np.ndarray, knee angle w.r.t knee joint.
		l1, float length of thign.
		l2, float length of shank

	Return:
		x coordinate of hip, y coordinate of hip, x coordinate of knee, 
		y coordinate of knee, x cooridnate of ankle, y coordinate of ankle. 
		In np.ndarray or float.

	"""
    if isinstance(theta1, float):
        cos = math.cos
        sin = math.sin
        xh = 0
        yh = 0
    else:
        cos = np.cos
        sin = np.sin
        xh = np.zeros(theta1.shape[0])
        yh = np.zeros(theta2.shape[0])

    xk = l1 * cos(theta1)
    yk = l1 * sin(theta1)
    xf = l1 * cos(theta1) + l2 * cos(theta1 + theta2)
    yf = l1 * sin(theta1) + l2 * sin(theta1 + theta2)

    return xh, yh, xk, yk, xf, yf


def ankle_joint(x_r, x_l, z_r, z_l, l1, l2, angle_offset=0):
    """Convert ankle to joint coordinates."""
    thetah_r, thetak_r = cart_joint(x_r, z_r, l1, l2, if_debug=False)
    thetah_l, thetak_l = cart_joint(x_l, z_l, l1, l2, if_debug=False)

    return (
        thetah_r + angle_offset,
        thetah_l + angle_offset,
        thetak_r + angle_offset,
        thetak_l + angle_offset,
    )


def joint_ankle(thetah_r, thetah_l, thetak_r, thetak_l, l1, l2, angle_offset=0):
    """Convert joint to ankle coordinates."""
    _, _, _, _, x_r, z_r = joint_cart(
        thetah_r - angle_offset, thetak_r - angle_offset, l1, l2
    )
    _, _, _, _, x_l, z_l = joint_cart(
        thetah_l - angle_offset, thetak_l - angle_offset, l1, l2
    )

    return x_r, x_l, z_r, z_l


def d_2_r(theta):
    """From degree to radial. Float or np.ndarray"""
    if isinstance(theta, float):
        return theta * pi / 180
    else:
        return theta * np.pi / 180


def r_2_d(theta):
    """From raidal to degree. float or np.ndarray"""
    if isinstance(theta, float):
        return theta * 180 / pi
    else:
        return theta * 180 / np.pi
