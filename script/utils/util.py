import csv
import math
from copy import deepcopy

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt

from .converter import *


def ideal_body_p(h=None, w=None):
    """Get the ideal body parameters.

    The parameters are calculated according to a textbook. If any arguments are
    not given, the corresponding body parameters will not be calculated.

    Args:
        h: The total height of the body (head to foot).
        w: the total weight of the body.

    Returns:
        A dict mapping keys to corresponding values.
    """
    body_para = {}
    if h is not None:
        body_height = {
            "height": h,
            "head_len": 0.13 * h,
            "eye_height": 0.936 * h,
            "shoulder_width": 0.259 * h,
            "arm_len": 0.186 * h,
            "forearm_len": 0.146 * h,
            "hand_len": 0.108 * h,
            "breast_height": 0.72 * h,
            "body_width": 0.174 * h,
            "hip_width": 0.191 * h,
            "hip_height": 0.53 * h,
            "thign_len": (0.53 - 0.285) * h,
            "shank_len": (0.285 - 0.039) * h,
            "thign_COM_coe": 0.433,
            "shank_COM_coe": 0.433,
            "thign_COM": 0.433 * (0.53 - 0.285) * h,
            "shank_COM": 0.433 * 0.53 * h,
            "COM_height": 0.53 * h,
            "ankle_len": 0.039 * h,
            "foot_width": 0.055,
            "foot_len": 0.152 * h,
        }
        body_para = {**body_para, **body_height}

    if w is not None:
        body_weight = {"weight": w, "thign_weight": 0.1 * w, "shank_weight": 0.465 * w}
        body_para = {**body_para, **body_weight}

    return body_para


def smooth_interp(rough_num, fine_num, y):
    """Smooth give data by doing interpolating

    Interpolate between any given points, so the curve is more continuous. The
    X is pre-determined by a sequence from 0 to 1, which should not affect the 
    result. The interpolation method is 'cubic'. The data points are increased.

    Args:
        rough_num: The number of data points before interpolation.
        fine_num: The number of data points after interpolation.
        y: The data needed to be smoothed of np.array type.

    Returns:
        An array of smoothed data.

    """
    X = np.linspace(0, 1, rough_num)
    F = interp1d(X, y, kind="cubic")

    X_new = np.linspace(0, 1, fine_num)
    return F(X_new)


def smooth_data_aver(num_gram, y):
    """Smooth data by averaging in a n-gram manner.

    y(i) = sum(y(i-j))|j=-k->j=k / (2*k+1), where k is the num_gram

    Args:
        num_gram: How many points should be averaged on for each side.
        y: Array of data needed to be smoothed.

    Returns:
        Array of smoothed data.

    """
    for i, val in enumerate(y.tolist()):
        if i < num_gram:
            continue

        if i > y.shape[0] - num_gram:
            break

        temp = val
        for j in range(num_gram):
            temp += y[i + j]
            temp += y[i - j]

        y[i] = temp / (num_gram * 2 + 1)

    return y


def smooth_data_aver_sing(k, y):
    """Smooth data by averaging in only one side.

    y(i) = sum(y(i-j))|j=-(k-1)->j=0 / k.

    Args:
        k: How many points should be averaged.
        y: 1D Array of data needed to be smoothed.

    Returns:
        Array of smoothed data.

    """
    for i, _ in enumerate(y.tolist()):
        if i < k:
            continue

        temp = 0
        for j in range(k):
            temp += y[i - j]

        y[i] = temp / k

    return y


def fft_ana(data, interval, xlim=None, title=None, plot=False):
    """Analyze data by fft. title is for plotting."""
    sp = np.fft.fft(data - np.mean(data))
    freq = np.fft.fftfreq(data.shape[0], interval)
    if plot:
        fig, ax = plt.subplots()
        ax.set_title("right knee")
        ax.stem(freq, abs(sp))
        if xlim is not None:
            ax.set_xlim(*xlim)
        if title is not None:
            ax.set_title(title)
        ax.set_xlabel("Frequency in Hertz [Hz]")
        ax.set_ylabel("Frequency Domain (Spectrum) Magnitude")

    return sp, freq


def ifft_ana(_w, freq, cutoff, plot=False, orig_data=None):
    """Inverse fft"""
    w = deepcopy(_w)
    for i, f in enumerate(freq):
        if abs(f) > cutoff:
            w[i] = 0 + 0j

    ival = np.fft.ifft(w)
    if plot:
        plt.figure()
        plt.plot(ival, lw=2)
        plt.plot(orig_data - np.mean(orig_data), "--", lw=2)
        plt.title(f"Inverse FFT with Cutoff of {cutoff} Hz")
        plt.legend(["Inverse FFT", "Original"])
        plt.xlabel("Samples")
        plt.ylabel("Val")

    return np.fft.ifft(w)


def butterworth_filt_low_data(data, cutoff, freq, order):
    """Filtering the data using butterworth low-pass filter."""
    nyq = freq * 0.5
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype="low", analog=False)
    return filtfilt(b, a, data)


def butterworth_filt_low_dict(data_dict, keys, cutoff, freq, order):
    """Same to low_data but for dictionaries."""
    output_dict = {}
    for k, v in data_dict.items():
        if k in keys:
            output_dict[k] = butterworth_filt_low_data(v, cutoff, freq, order)
    return output_dict


def gait_data_entry(row_num):
    """Switch implementation to get key w.r.t different row_num"""
    entry = {
        1: "time",
        2: "r_ankle_local_x",
        3: "r_ankle_local_y",
        4: "r_ankle_local_z",
        5: "r_ankle_global_x",
        6: "r_ankle_global_y",
        7: "r_ankle_global_z",
        8: "r_knee_local_x",
        9: "r_knee_local_y",
        10: "r_knee_local_z",
        11: "r_knee_global_x",
        12: "r_knee_global_y",
        13: "r_knee_global_z",
        14: "r_knee_joint",
        15: "r_hip_local_x",
        16: "r_hip_local_y",
        17: "r_hip_local_z",
        18: "r_hip_global_x",
        19: "r_hip_global_y",
        20: "r_hip_global_z",
        21: "r_hip_joint",
        22: "l_ankle_local_x",
        23: "l_ankle_local_y",
        24: "l_ankle_local_z",
        25: "l_ankle_global_x",
        26: "l_ankle_global_y",
        27: "l_ankle_global_z",
        28: "l_knee_local_x",
        29: "l_knee_local_y",
        30: "l_knee_local_z",
        31: "l_knee_global_x",
        32: "l_knee_global_y",
        33: "l_knee_global_z",
        34: "l_knee_joint",
        35: "l_hip_local_x",
        36: "l_hip_local_y",
        37: "l_hip_local_z",
        38: "l_hip_global_x",
        39: "l_hip_global_y",
        40: "l_hip_global_z",
        41: "l_hip_joint",
        42: "r_hs",
        43: "r_to",
        44: "l_hs",
        45: "l_to",
    }

    return entry.get(row_num, "wrong row number!")


def load_gait_data(path):
    """Load data w.r.t /data/fsys.md regulation."""
    data = {}
    with open(path, "r") as file:
        reader = csv.reader(file)
        for i, row in enumerate(reader):
            if i == 0:
                continue

            if i < 42:
                temp = np.array([float(val) for val in row])
                key = gait_data_entry(i)
            else:
                temp = np.array([True if val == "True" else False for val in row])
                key = gait_data_entry(i)

            data[key] = temp

    return data


def check_leg_lengths(path=None, data=None, local=True):
    """Check the length of legs

    Check the length of legs with given data. Will check thign, shank lengths of
    right and left legs. (l1 for thign and l2 for shank)

    Args:
        path: Path to data file (won't be used when given data arg)
        data: Gait data defined in the doc folder.
        local: True for w.r.t local frame, False for w.r.t global frame.

    Returns:
        dict of length change. 'r_l1_max' is the max value of right thign length.
        Available: r_l1_max, r_l1_min, r_l1_range.

    """
    if data is None:
        data = load_gait_data(path)

    if local:
        scope = "_local_"
    else:
        scope = "_global_"

    r_ankle_x = data["r_ankle" + scope + "x"]
    r_ankle_y = data["r_ankle" + scope + "y"]
    r_ankle_z = data["r_ankle" + scope + "z"]
    r_knee_x = data["r_knee" + scope + "x"]
    r_knee_y = data["r_knee" + scope + "y"]
    r_knee_z = data["r_knee" + scope + "z"]
    r_hip_x = data["r_hip" + scope + "x"]
    r_hip_y = data["r_hip" + scope + "y"]
    r_hip_z = data["r_hip" + scope + "z"]

    l_ankle_x = data["l_ankle" + scope + "x"]
    l_ankle_y = data["l_ankle" + scope + "y"]
    l_ankle_z = data["l_ankle" + scope + "z"]
    l_knee_x = data["l_knee" + scope + "x"]
    l_knee_y = data["l_knee" + scope + "y"]
    l_knee_z = data["l_knee" + scope + "z"]
    l_hip_x = data["l_hip" + scope + "x"]
    l_hip_y = data["l_hip" + scope + "y"]
    l_hip_z = data["l_hip" + scope + "z"]

    r_l2 = np.average(
        np.sqrt(
            np.power(r_ankle_x - r_knee_x, 2)
            + np.power(r_ankle_y - r_knee_y, 2)
            + np.power(r_ankle_z - r_knee_z, 2)
        )
    )

    r_l1 = np.average(
        np.sqrt(
            np.power(r_knee_x - r_hip_x, 2)
            + np.power(r_knee_y - r_hip_y, 2)
            + np.power(r_knee_z - r_hip_z, 2)
        )
    )

    r_l1_max = np.max(r_l1)
    r_l2_max = np.max(r_l2)
    r_l1_min = np.min(r_l1)
    r_l2_min = np.min(r_l2)
    r_l1_range = r_l1_max - r_l1_min
    r_l2_range = r_l2_max - r_l2_min
    r_l1_mean = np.mean(r_l1)
    r_l2_mean = np.mean(r_l2)

    l_l2 = np.average(
        np.sqrt(
            np.power(l_ankle_x - l_knee_x, 2)
            + np.power(l_ankle_y - l_knee_y, 2)
            + np.power(l_ankle_z - l_knee_z, 2)
        )
    )

    l_l1 = np.average(
        np.sqrt(
            np.power(l_knee_x - l_hip_x, 2)
            + np.power(l_knee_y - l_hip_y, 2)
            + np.power(l_knee_z - l_hip_z, 2)
        )
    )

    l_l1_max = np.max(l_l1)
    l_l2_max = np.max(l_l2)
    l_l1_min = np.min(l_l1)
    l_l2_min = np.min(l_l2)
    l_l1_range = l_l1_max - l_l1_min
    l_l2_range = l_l2_max - l_l2_min
    l_l1_mean = np.mean(l_l1)
    l_l2_mean = np.mean(l_l2)

    return {
        "r_l1_max": r_l1_max,
        "r_l2_max": r_l2_max,
        "r_l1_min": r_l1_min,
        "r_l2_min": r_l2_min,
        "r_l1_range": r_l1_range,
        "r_l2_range": r_l2_range,
        "r_l1_mean": r_l1_mean,
        "r_l2_mean": r_l2_mean,
        "l_l1_max": l_l1_max,
        "l_l2_max": l_l2_max,
        "l_l1_min": l_l1_min,
        "l_l2_min": l_l2_min,
        "l_l1_range": l_l1_range,
        "l_l2_range": l_l2_range,
        "l_l1_mean": l_l1_mean,
        "l_l2_mean": l_l2_mean,
    }


def plot_gait_ankle(
    x_r, x_l, z_r, z_l, time=None, color_x="slategrey", color_z="orangered"
):
    """Plot ankles' trajectory.

    Plot ankles trjectories of both legs. Need call show() outside of this 
    function.The unit for displacement is mm and for time is second.

    Args:
        time: The time stamps in second. Set to None will set the xlabel to 
        'sample'.
        x_r: X coordinate of right ankle.
        x_l: X coordinate of left ankle.
        z_r: Z coordinate of right ankle.
        z_l: Z coordinate of left ankle.
        color_x: color for x coordinates. default is slategray.
        color_z: color for z coordinates. default is orangered.

    """
    _, ax1 = plt.subplots()
    if time is None:
        line1 = ax1.plot(x_r, color=color_x, lw=2)
        line2 = ax1.plot(x_l, "--", color=color_x, lw=2)
        ax1.set_xlabel("Samples")
        ax2 = ax1.twinx()
        line3 = ax2.plot(z_r, color=color_z, lw=2)
        line4 = ax2.plot(z_l, "--", color=color_z, lw=2)
    else:
        line1 = ax1.plot(time, x_r, color=color_x, lw=2)
        line2 = ax1.plot(time, x_l, "--", color=color_x, lw=2)
        ax1.set_xlabel("time (s)")
        ax2 = ax1.twinx()
        line3 = ax2.plot(time, z_r, color=color_z, lw=2)
        line4 = ax2.plot(time, z_l, "--", color=color_z, lw=2)

    ax1.set_title("ankle trajectory")
    ax1.set_ylabel("x displacement of ankle (mm)", color=color_x)
    ax2.set_ylabel("z displacement of ankle (mm)", color=color_z)
    lines = line1 + line2 + line3 + line4
    ax1.legend(lines, ["right x", "left x", "right z", "left z"])


def plot_gait_ankle_2(
    x_r, x_l, z_r, z_l, time=None, color_x="slategrey", color_z="orangered"
):
    """Plot ankles' trajectory on two figures.

    Plot ankles trjectories of both legs. Need call show() outside of this 
    function. The unit for displacement is mm and for time is second. The first 
    figure will plot right leg and the second figure will plot left leg.

    Args:
        time: The time stamps in second. Set to None will set the xlabel to 
        'sample'.
        x_r: X coordinate of right ankle.
        x_l: X coordinate of left ankle.
        z_r: Z coordinate of right ankle.
        z_l: Z coordinate of left ankle.
        color_x: color for x coordinates. default is slategray.
        color_z: color for z coordinates. default is orangered.

    """
    _, ax1 = plt.subplots()
    if time is None:
        line1 = ax1.plot(x_r, color=color_x, lw=2)
        ax1.set_xlabel("Samples")
        ax2 = ax1.twinx()
        line2 = ax2.plot(z_r, color=color_z, lw=2)
    else:
        line1 = ax1.plot(time, x_r, color=color_x, lw=2)
        ax1.set_xlabel("time (s)")
        ax2 = ax1.twinx()
        line2 = ax2.plot(time, z_r, color=color_z, lw=2)

    ax1.set_title("right ankle trajectory")
    ax1.set_ylabel("x displacement of ankle (mm)", color=color_x)
    ax2.set_ylabel("z displacement of ankle (mm)", color=color_z)
    lines = line1 + line2
    ax1.legend(lines, ["right x", "right z"])

    _, ax1 = plt.subplots()
    if time is None:
        line1 = ax1.plot(x_l, color=color_x, lw=2)
        ax1.set_xlabel("Samples")
        ax2 = ax1.twinx()
        line2 = ax2.plot(z_l, color=color_z, lw=2)
    else:
        line1 = ax1.plot(time, x_l, color=color_x, lw=2)
        ax1.set_xlabel("time (s)")
        ax2 = ax1.twinx()
        line2 = ax2.plot(time, z_l, color=color_z, lw=2)

    ax1.set_title("left ankle trajectory")
    ax1.set_ylabel("x displacement of ankle (mm)", color=color_x)
    ax2.set_ylabel("z displacement of ankle (mm)", color=color_z)
    lines = line1 + line2
    ax1.legend(lines, ["left x", "left z"])


def cal_link_com(
    x1, x2, c, plot=False, titles=None, x_labels=None, y_labels=None, legends=None
):
    """Function used to calculate trajectory of CoM in one direction.
    
    For example, refer to bodyM.get_link_dyn()

    Args:
        x1: List of arrays of coordinates of point1.
        x2: List of arrays of coordinates of point2.
        c: List of ratio calculated from x1. x_i = x1_i + c_i * (x2_i - x1_i)
        plot: If plot the trajectories of x1, x2 and com. Default False.
        titles: List of titles. Default None.
        x_labels: List of x labels. Default None.
        y_labels: List of y labels. Default None.
        legends: List of legends with order: x1, x, x2. Default None.

    Return:
        List of arrays of coordinates

    """
    x = []
    for i, _ in enumerate(x1):
        x.append(x1[i] + c[i] * (x2[i] - x1[i]))

        if plot:
            plt.figure()
            plt.plot(x1[i])
            plt.plot(x[i])
            plt.plot(x2[i])

            if titles is not None:
                plt.title(titles[i])
            if x_labels is None:
                plt.xlabel(x_labels[i])
            if y_labels is not None:
                plt.ylabel(y_labels[i])
            if legends is not None:
                plt.legend(legends[i])
            plt.show()

    return x


def cal_link_angular_moment(x, y, z, dx, dy, dz, m):
    """Calculate the angular momentum of a link w.r.t origin point"""
    r = [[x[i], y[i], z[i]] for i in range(x.shape[0])]
    r = np.array(r)

    p = [[dx[i], dy[i], dz[i]] for i in range(x.shape[0])]
    p = m * np.array(p)

    L = np.cross(r, p)

    L_dot = np.gradient(L, 0.001, axis=0)

    return L, L_dot


def save_csv(path, row_col, data_list):
    """Write data into csv file w.r.t MatrixIo's format.

        Args:
            path: File path.
            row_col: (row_number, col_number). Deprecated.
            data_list: List of data to be written. Each row is a element.
    
    """
    N_row = len(data_list)
    if isinstance(data_list[0], float):
        N_col = 1
    else:
        N_col = len(data_list[0])
    row_col = (N_row, N_col)
    with open(path, mode="w", newline="") as data_file:
        writer = csv.writer(data_file, delimiter=",")
        data_file.write(f"#size: {row_col[0]} {row_col[1]}\n")
        for data in data_list:
            if isinstance(data, float):
                writer.writerow([data])
            else:
                writer.writerow(data)


def gait_check_triangle(l1, l2, path=None, data=None):
    """Check if a gait pattern is valid.
    
        Check if a gait pattern is valid by checking if pattern could form a 
        valid triangle.

        Args:
            path, String path to gait pattern data. It is ignored whenever data
            is given.
            data, Data file w.r.t document.
            l1, Thign length.
            l2, Shank length.
    """
    if data is None:
        data = load_gait_data(path)

    flag = False
    r_x = data["r_ankle_local_x"]
    l_x = data["l_ankle_local_x"]
    r_z = data["r_ankle_local_z"]
    l_z = data["l_ankle_local_z"]

    r_d = np.sqrt(np.power(r_x, 2) + np.power(r_z, 2))
    l_d = np.sqrt(np.power(l_x, 2) + np.power(l_z, 2))

    l12 = l1 + l2
    d1 = r_d + l1
    d2 = r_d + l2
    if np.greater(r_d, l12).any():
        flag = True
        print(
            "Distance from right ankle to origin is bigger than sumation of"
            " thign and shank length!"
        )
    if np.greater(l2, d1).any():
        flag = True
        print(
            "Right Shank length is bigger than summation of d_a_o and thign" " length!"
        )
    if np.greater(l1, d2).any():
        flag = True
        print(
            "Right thign length is bigger than summation of d_a_o and shank" " length!"
        )

    d1 = l_d + l1
    d2 = l_d + l2
    if np.greater(l_d, l12).any():
        flag = True
        print(
            "Distance from left ankle to origin is bigger than sumation of"
            " thign and shank length!"
        )
    if np.greater(l2, d1).any():
        flag = True
        print(
            "Left Shank length is bigger than summation of d_a_o and thign" " length!"
        )
    if np.greater(l1, d2).any():
        flag = True
        print(
            "Left thign length is bigger than summation of d_a_o and shank" " length!"
        )

    if flag:
        raise ValueError("Gait pattern is wrong!")
    else:
        print("triangle test passed!")


def segment_angle(x, z):
    """Caculate the angle.
    
        One point is at origin. One segment is along the y/z axis.

    """
    angle = np.arctan2(z, x)
    return r_2_d(angle) + 90


def regress_slope(y, kernel_size, x=None):
    """Using regression to get slope

    In 1-D manner. It's basically a weighted conv1d / time. Sphere correction is
    ignored.

    Args:
        y, Array that needs to calculate slope. D X T.
        kernerl_size, The kernel size used to regression. Should be odd.
        x, Predefined time interval. Default is None for automatical generation.

    Returns:
        Slope of dimension: D X (T - kernel_size * 2)
    """
    kernel_side_len = (kernel_size - 1) / 2
    D, W = y.shape
    r = np.zeros((D, W - kernel_size - 1))

    if x is None:
        x = np.array(range(-kernel_side_len, kernel_side_len + 1))

    for i in range(W - kernel_size + 1):
        for d in range(D):
            r[d, i] = _regress_slope(y[d, i : i + kernel_size], x)

        r[:, i] /= np.linalg.norm(r[:, i])

    return r


def _regress_slope(y, x):
    """Implementation of slope."""
    return np.dot(y - np.mean(y), x) / np.dot(x, x)


def gen_unit(g):
    """Plot points on a unit circle w.r.t gait phase."""
    return np.array([np.cos(g), np.sin(g)]).T


def v_angle(v1, v2):
    """Angle between vector v1 and v2"""
    temp = np.zeros(v1.shape[0])
    for i in range(v1.shape[0]):
        temp[i] = np.dot(v1[i, :], v2[i, :])
    return np.arccos((temp / (np.linalg.norm(v1, axis=1) * np.linalg.norm(v2, axis=1))))


def get_status_2(
    data_dict, thd=0.3, thcopr=0.7, thcopl=0.7, l1=0.4, l2=0.41, dt=0.04, tuning=False
):
    """Determine the walking status w.r.t distance and pressure sumation 
    gradient.

    Gradient of distance determines the start of walking and gradient of cop 
    summation determines the end of walking.

    Args:
        data_dict, Dictionary contains the needed, data.
        thd, Threshold for distance gradient.
        thcopr, Threshold for right leg's pressure summation.
        thcopl, Threshold for left leg's pressure summation.
        l1, Length of thigh.
        l2, Length of shank.
        plot, If plot the gradients.
    
    Returns:
        A array indicates the status (0 for pausing and 1 for walking) (length 
        of the array is 1 step shorter than the data).
    """
    status = np.zeros(data_dict["status"].shape[0] - 1)
    rh = d_2_r(data_dict["right_hip"] - 90)
    lh = d_2_r(data_dict["left_hip"] - 90)
    rk = d_2_r(data_dict["right_knee"] - 90)
    lk = d_2_r(data_dict["left_knee"] - 90)

    _, _, _, _, xr, _ = joint_cart(rh, rk, l1, l2)
    _, _, _, _, xl, _ = joint_cart(lh, lk, l1, l2)

    d = xr - xl
    d = (d - np.mean(d)) / np.std(d)
    d_g = np.abs(np.gradient(d, 0.04))

    cop_right = data_dict["cop_right_sum"]
    cop_left = data_dict["cop_left_sum"]
    cop_r_g = np.gradient(cop_right, dt)
    cop_l_g = np.gradient(cop_left, dt)

    walking = False
    right = False
    init = False  # Used to ignore noise before the command is sent.
    stat = 0
    count = 0
    for i in range(len(data_dict["status"]) - 1):
        if init is False:
            if data_dict["status"][i] > 0:
                if data_dict["status"][i] == 1:
                    right = False
                else:
                    right = True
                init = True
            else:
                continue

        if walking is False:
            if d_g[i] > thd:
                walking = True
                stat = 1
        elif right is True and data_dict["status"][i + 1] == 0:
            if cop_r_g[i] > thcopr:
                walking = False
                right = False
                stat = 0
        else:
            if cop_l_g[i] > thcopl and data_dict["status"][i + 1] == 0:
                walking = False
                right = True
                stat = 0

        status[i] = stat

    if tuning:
        plt.figure()
        plt.title("Segmentation of Gait")
        # plt.plot(data_dict['right_hip'], lw=2)
        # plt.plot(data_dict['left_hip'], lw=2)
        plt.plot(d, lw=2)
        plt.plot(cop_right, lw=2)
        plt.plot(cop_left, lw=2)
        plt.plot(data_dict["status"][1:], lw=2)
        plt.plot(status, lw=2)
        # plt.legend(['right_hip', 'left_hip', 'Speretaing Result'])
        plt.grid(True)
        plt.legend(
            ["distance", "cop_right", "cop_left", "status", "segmentation result"]
        )
        plt.ylabel("value after standization")
        plt.xlabel("# sample")

        plt.figure()
        plt.plot(cop_right, lw=2)
        plt.plot(cop_left, lw=2)
        plt.plot(data_dict["status"])
        plt.legend(["right", "left", "original_status"])

        plt.figure()
        plt.title("gradient of distance")
        plt.plot(d_g, lw=2)
        plt.plot([0, len(d_g)], [thd, thd], lw=2)
        plt.plot(data_dict["status"] / 1, lw=2)
        plt.legend(["gradient of distance", "threshold", "rough status"])
        plt.xlabel("samples")
        plt.ylabel("gradient value")

        plt.figure()
        plt.title("gradient of pressure")
        plt.plot(cop_r_g, lw=2)
        plt.plot(cop_l_g, lw=2)
        plt.plot([0, len(d_g)], [thcopr, thcopr], lw=3)
        plt.plot(data_dict["status"], lw=2)
        plt.plot([0, len(d_g)], [thcopl, thcopl], lw=3)
        plt.legend(["right foot", "left foot", "threshold", "rough status"])
        plt.xlabel("samples")
        plt.ylabel("gradient value")

    return status


def get_status(diff, threshold, consecutive_count, status_input):
    """Get gait steps by threshold and given difference.
    
    consecutive_count is the minimum required number of excedding the therehold.
    """
    if diff.ndim > 1:
        diff.reshape(-1)

    status = np.zeros(diff.shape)

    start_index = np.argwhere(status_input > 0)[0].item()
    stat_input = status_input[start_index:]

    counter = 0
    stat = 0
    for i, val in enumerate(diff[start_index:]):
        # if stat_input[i] > 0:
        # 	stat = 1
        # 	counter = 0
        if ((val > threshold) and (stat == 0)) or ((val < threshold) and (stat == 1)):
            if counter > consecutive_count - 1:
                counter = 0
                stat = 1 - stat
            else:
                counter += 1
        else:
            counter = 0

        status[i + start_index] = stat

    return status


def slice_gait_phase(data_dict, start, end):
    """Slicing the data w.r.t given start and end index."""
    dict_slice = {}
    for k, v in data_dict.items():
        dict_slice[k] = v[start:end]

    return dict_slice


def plot_gait_phase(data_dict, keys, status=None, start=None, end=None):
    """Plot the gait phase data.

    Args:
        data_dict, Dictionary contains the data.
        keys, Needed features.
        status, Calculated status, if None will not plot.
        start, From which index to start ploting the data.
        end, Till which index to end ploting the data.
    """
    if start is not None:
        data_dict = slice_gait_phase(data_dict, start, end)

    plt.figure()
    legends = []

    for k in data_dict.keys():
        if k in keys:
            plt.plot(data_dict[k], lw=2)
            legends.append(k)

    plt.plot(data_dict["status"], lw=2)
    legends.append("status")
    plt.legend(legends)
    plt.title("gait phase data")
    plt.xlabel("sample")
    plt.ylabel("value")

    plt.figure()
    plt.plot(data_dict["right_hip"], lw=2)
    plt.plot(data_dict["left_hip"], lw=2)
    plt.title("HIP")
    plt.legend(["r", "l"])

    plt.figure()
    plt.plot(data_dict["right_knee"], lw=2)
    plt.plot(data_dict["left_knee"], lw=2)
    plt.title("Knee")
    plt.legend(["r", "l"])


def aver_smooth_data_dict(data_dict, aver, keys):
    """Averaging given data in the dictionary w.r.t give keys and kernel size.
    """
    for k, v in data_dict.items():
        if k in keys:
            data_dict[k] = smooth_data_aver_sing(aver, v)


def std_data_dict(data_dict, keys):
    """Standarze given data in the dictionary w.r.t give keys."""
    kl, meanl, stdl = [], [], []
    sta = {
        "cop_left_x": [9.38218919497052, 3.6970596515271086],
        "cop_left_y": [2.3028563467944063, 0.3757744981335798],
        "cop_left_sum": [7829.065097256898, 4527.2985391067905],
        "cop_right_x": [6.913140531055664, 2.179051662942722],
        "cop_right_y": [3.2382264161766665, 0.65744621382958],
        "cop_right_sum": [16740.71911779808, 5854.201498372955],
        "right_hip": [26.084104747756545, 16.965798002077705],
        "left_hip": [22.242233053497838, 18.453476864782658],
        "right_knee": [22.750198412078124, 19.39906921864179],
        "left_knee": [15.814045834678847, 18.241475483315313],
    }
    for k, v in data_dict.items():
        if k in keys:
            data_dict[k] = (v - np.mean(v)) / np.std(v)
            # data_dict[k] = (v - sta[k][0]) / sta[k][1]
            # kl.append(k)
            # meanl.append(np.mean(v))
            # stdl.append(np.std(v))
    # with open("tmp/data_statistics.txt", "w") as f:
    #     for k, m, s in zip(kl, meanl, stdl):
    #         f.write(f"{k}:    {m}    {s}\n")


def scale_data_dict(data_dict, keys):
    """Standarze given data in the dictionary w.r.t give keys."""
    range_dict = {
        "hip": (-22, 90),
        "knee": (-1, 90),
        "x": (1, 15),
        "y": (1, 7),
        "sum": (0, 3e4),
    }
    for k, v in data_dict.items():
        if k in keys:
            temp = k.split("_")[-1]
            low, high = range_dict[temp][0], range_dict[temp][1]
            mean = (high - low) / 2
            # data_dict[k] = (v - low) / (high - low)
            data_dict[k] = (v - mean) / (high - low) * 2
            # low, high = np.min(v), np.max(v)
            # print(f"{k}: {low}, {high}")
            # mean = (high - low) / 2
            # data_dict[k] = (v - mean) / (high - low) * 2


def load_gait_phase_data_old(
    path,
    keys=None,
    plot=False,
    stand=False,
    scale=False,
    status=False,
    aver_len=0,
    raw=False,
    **kwargs,
):
    """Load gait phase data as a dictionary.
    
    Args:
        path, Path to the data file.
        keys, List of features needed. If not specified then all the features 
        are used.
        plot, Option to plot the data.
        stand, If standardize the data to zero mean and 1 standard deviation.
        status, If calculate the status.
        aver_len, Kernel size used in the average filter. Default is 0 which 
        means no filtering.
        raw, If return raw data.

    Returns:
        A dictionary of data.
        A n-d array composed of features specified in keys.
    """
    data = np.loadtxt(path)
    data_len = data.shape[0]

    # assert data.shape[1] == 36, "The column number is not 17!"

    data_dict = {}
    data_dict["cop_time"] = data[:, 0]
    data_dict["exo_time"] = data[:, 7]

    data_dict["right_traj"] = data[:, 12]
    data_dict["left_traj"] = data[:, 13]
    data_dict["torso_front"] = data[:, 14]
    # data_dict['torso_side'] = data[:, 15]
    data_dict["status"] = data[:, 15]

    data_dict["cop_left_x"] = data[:, 1]
    data_dict["cop_left_y"] = data[:, 2]
    data_dict["cop_left_sum"] = data[:, 3]
    data_dict["cop_right_x"] = data[:, 4]
    data_dict["cop_right_y"] = data[:, 5]
    data_dict["cop_right_sum"] = data[:, 6]

    data_dict["right_hip"] = data[:, 9]
    data_dict["left_hip"] = data[:, 10]
    data_dict["right_knee"] = data[:, 11]
    data_dict["left_knee"] = data[:, 12]

    # data_dict["right_hip"] = data[:, 8]
    # data_dict["left_hip"] = data[:, 9]
    # data_dict["right_knee"] = data[:, 10]
    # data_dict["left_knee"] = data[:, 11]

    # plt.figure()
    # plt.plot(data_dict["right_hip"])
    # plt.plot(data_dict["left_hip"])
    # plt.plot(data_dict["right_knee"])
    # plt.plot(data_dict["left_knee"])
    # plt.legend(["rh", "lh", "rk", "lk"])
    # plt.show()

    if raw:
        data_dict_raw = deepcopy(data_dict)

    if aver_len:
        _keys = deepcopy(keys)
        # _keys.extend(['cop_left_sum', 'cop_right_sum'])
        aver_smooth_data_dict(data_dict, aver_len, _keys)

    if scale:
        _keys = deepcopy(keys)
        # _keys.extend(['cop_left_sum', 'cop_right_sum'])
        scale_data_dict(data_dict, _keys)

    if stand:
        _keys = deepcopy(keys)
        # _keys.extend(['cop_left_sum', 'cop_right_sum'])
        std_data_dict(data_dict, _keys)

    if keys == None:
        keys = data_dict.keys()

    if status:
        stat = get_status_2(data_dict, **kwargs)
    else:
        stat = None

    data = np.zeros((data_len, len(keys)))
    for i, key in enumerate(keys):
        data[:, i] = data_dict[key]

    if plot:
        plot_gait_phase(data_dict, keys, status=stat)
        # plot_gait_phase(data_dict, keys, status=stat, start=3000, end=11000)

    if status:
        if raw:
            return data_dict, data, stat, data_dict_raw
        else:
            return data_dict, data, stat
    else:
        if raw:
            return data_dict, data, data_dict_raw
        else:
            return data_dict, data


def load_gait_phase_data(
    path,
    keys=None,
    plot=False,
    stand=False,
    scale=False,
    status=False,
    aver_len=0,
    raw=False,
    **kwargs,
):
    """Load gait phase data as a dictionary.
    
    Args:
        path, Path to the data file.
        keys, List of features needed. If not specified then all the features 
        are used.
        plot, Option to plot the data.
        stand, If standardize the data to zero mean and 1 standard deviation.
        status, If calculate the status.
        aver_len, Kernel size used in the average filter. Default is 0 which 
        means no filtering.
        raw, If return raw data.

    Returns:
        A dictionary of data.
        A n-d array composed of features specified in keys.
    """
    data = np.loadtxt(path)
    data_len = data.shape[0]

    assert data.shape[1] == 36, "The column number is not 36!"

    data_dict = {}
    data_dict["time"] = data[:, 0]
    data_dict["right_hip"] = data[:, 1]
    data_dict["left_hip"] = data[:, 2]
    data_dict["right_knee"] = data[:, 3]
    data_dict["left_knee"] = data[:, 4]

    data_dict["cop_right_x"] = data[:, 5]
    data_dict["cop_left_x"] = data[:, 6]
    data_dict["cop_right_y"] = data[:, 7]
    data_dict["cop_left_y"] = data[:, 8]
    data_dict["cop_right_sum"] = data[:, 9]
    data_dict["cop_left_sum"] = data[:, 10]
    data_dict["status"] = data[:, 11]

    data_dict["gp_cal"] = data[:, 12]
    data_dict["gp_actual"] = data[:, 13]

    data_dict["right_hip_vel"] = data[:, 14]
    data_dict["left_hip_vel"] = data[:, 15]
    data_dict["right_knee_vel"] = data[:, 16]
    data_dict["left_knee_vel"] = data[:, 17]

    data_dict["right_hip_acc"] = data[:, 18]
    data_dict["left_hip_acc"] = data[:, 19]
    data_dict["right_knee_acc"] = data[:, 20]
    data_dict["left_knee_acc"] = data[:, 21]

    data_dict["right_hip_cmd"] = data[:, 22]
    data_dict["left_hip_cmd"] = data[:, 23]
    data_dict["right_knee_cmd"] = data[:, 24]
    data_dict["left_knee_cmd"] = data[:, 25]

    data_dict["right_hip_cmd_vel"] = data[:, 26]
    data_dict["left_hip_cmd_vel"] = data[:, 27]
    data_dict["right_knee_cmd_vel"] = data[:, 28]
    data_dict["left_knee_cmd_vel"] = data[:, 29]

    data_dict["right_hip_cmd_acc"] = data[:, 30]
    data_dict["left_hip_cmd_acc"] = data[:, 31]
    data_dict["right_knee_cmd_acc"] = data[:, 32]
    data_dict["left_knee_cmd_acc"] = data[:, 33]

    data_dict["torsor_front"] = data[:, 34]
    data_dict["torsor_side"] = data[:, 35]

    if raw:
        data_dict_raw = deepcopy(data_dict)

    if aver_len:
        _keys = deepcopy(keys)
        # _keys.extend(['cop_left_sum', 'cop_right_sum'])
        aver_smooth_data_dict(data_dict, aver_len, _keys)

    if scale:
        _keys = deepcopy(keys)
        # _keys.extend(['cop_left_sum', 'cop_right_sum'])
        scale_data_dict(data_dict, _keys)

    if stand:
        _keys = deepcopy(keys)
        # _keys.extend(['cop_left_sum', 'cop_right_sum'])
        std_data_dict(data_dict, _keys)

    if keys == None:
        keys = data_dict.keys()

    if status:
        stat = get_status_2(data_dict, **kwargs)
    else:
        stat = None

    data = np.zeros((data_len, len(keys)))
    for i, key in enumerate(keys):
        data[:, i] = data_dict[key]

    if plot:
        plot_gait_phase(data_dict, keys, status=stat)
        # plot_gait_phase(data_dict, keys, status=stat, start=3000, end=11000)

    if status:
        if raw:
            return data_dict, data, stat, data_dict_raw
        else:
            return data_dict, data, stat
    else:
        if raw:
            return data_dict, data, data_dict_raw
        else:
            return data_dict, data


def save_gait_phase_data(data_dict, path=None):
    """Save gait phase data. Not used and outdated."""
    data_m = np.zeros((data_dict["cop_time"].shape[0], 17))
    data_m[:, 0] = data_dict["cop_time"]
    data_m[:, 1] = data_dict["cop_left_x"]
    data_m[:, 2] = data_dict["cop_left_y"]
    data_m[:, 3] = data_dict["cop_left_sum"]
    data_m[:, 4] = data_dict["cop_right_x"]
    data_m[:, 5] = data_dict["cop_right_y"]
    data_m[:, 6] = data_dict["cop_right_sum"]
    data_m[:, 7] = data_dict["exo_time"]
    data_m[:, 8] = data_dict["right_hip"]
    data_m[:, 9] = data_dict["left_hip"]
    data_m[:, 10] = data_dict["right_knee"]
    data_m[:, 11] = data_dict["left_knee"]
    data_m[:, 12] = data_dict["right_traj"]
    data_m[:, 13] = data_dict["left_traj"]
    data_m[:, 14] = data_dict["torso_front"]
    data_m[:, 15] = data_dict["torso_side"]
    data_m[:, 16] = data_dict["status"]

    if path is not None:
        np.savetext(path, data_m)

    return data_m


def seg_gait_phase(status):
    """segmentize the status array (1D).

    Args:
        status, 1D array of 0, 1 indicate walking and pausing.
    
    Returns:
        1D array of index while each is a start or end of one segment.
    """
    stat = status[0]
    index = [stat]
    for i, val in enumerate(status):
        if val != stat:
            if val == 0:
                index.append(i - 1)
            else:
                index.append(i)
            stat = val

    return np.array(index, dtype=np.int)


def _exo_ani(l1, l2, thetah_r, thetah_l, thetak_r, thetak_l, exo=True):
    """Gait Animation.
    
    Args:
        l1, Thigh length.
        l2, Shank length.
    """
    if exo:
        thetah_r = d_2_r(thetah_r - 90)
        thetah_l = d_2_r(thetah_l - 90)
        thetak_r = d_2_r(thetak_r - 90)
        thetak_l = d_2_r(thetak_l - 90)

    plt.figure()
    plt.plot(thetah_r)
    plt.plot(thetah_l)
    plt.plot(thetak_r)
    plt.plot(thetak_l)
    plt.legend(["thetah_r", "thetah_l", "thetak_r", "thetak_l"])
    # plt.show()

    xhr, zhr, xkr, zkr, xfr, zfr = joint_cart(thetah_r, thetak_r, l1, l2)
    xhl, zhl, xkl, zkl, xfl, zfl = joint_cart(thetah_l, thetak_l, l1, l2)

    N = xhr.shape[0]

    X = [[xfr[i], xkr[i], xhr[i], xhl[i], xkl[i], xfl[i]] for i in range(N)]
    Y = [[zfr[i], zkr[i], zhr[i], zhl[i], zkl[i], zfl[i]] for i in range(N)]

    def init():
        point_ani.set_data(X[0], Y[0])
        plt.axis([-2, 2, -2, 2])
        return (point_ani,)

    def update_points(i):
        point_ani.set_data(X[i], Y[i])
        plt.axis([-2, 2, -2, 2])
        return (point_ani,)

    num_frame = N
    fig = plt.figure(tight_layout=True)
    plt.xlabel("Y")
    plt.ylabel("Z")
    (point_ani,) = plt.plot(X[0], Y[0], lw=2)

    ani = animation.FuncAnimation(
        fig,
        update_points,
        np.arange(0, num_frame),
        init_func=init,
        interval=0,
        blit=True,
    )

    return ani


def exo_ani(l1, l2, data_dict):
    """Simulate exo's moving by recieved data."""
    torso = data_dict["torso_front_angle"]
    thetah_r = data_dict["right_hip"] - torso
    thetah_l = data_dict["left_hip"] - torso
    thetak_r = data_dict["right_knee"] - torso
    thetak_l = data_dict["left_knee"] - torso

    plt.figure()
    plt.plot(thetah_r)
    plt.plot(thetah_l)
    plt.plot(thetak_r)
    plt.plot(thetak_l)
    plt.legend(["thetah_r", "thetah_l", "thetak_r", "thetak_l"])
    # plt.show()

    ani = _exo_ani(l1, l2, thetah_r, thetah_l, thetak_r, thetak_l)

    return ani


def stand_gait_phase(_g_l):
    """Shift the starting phase to 0 and fix the range to [0, 2pi].

    Args:
        g_l, List, array or tensor.

    Return:
        Given data type.
    """
    if isinstance(_g_l, list):
        _g_l = np.array(_g_l)
        where = np.where
    elif isinstance(_g_l, np.ndarray):
        where = np.where
        zeros = np.zeros
        max = np.max
        abs = np.abs
    else:
        import torch

        where = torch.where
        zeros = torch.zeros
        max = torch.max
        abs = torch.abs

    g_l = _g_l - _g_l[0]
    g_l_diff = zeros(len(g_l))
    g_l_new = zeros(len(g_l))
    g_l_diff[1:] = g_l[1:] - g_l[:-1]
    g_l_diff = where(g_l_diff < -1, g_l_diff + 2 * math.pi, g_l_diff)
    # g_l_diff = where(g_l_diff >= 2 * math.pi, g_l_diff - 2 * math.pi, g_l_diff)

    for i in range(g_l_new.shape[0] - 1):
        g_l_new[i + 1] = g_l_new[i] + g_l_diff[i + 1]
    temp = deepcopy(g_l_new)

    # plt.figure()
    # plt.plot(_g_l, lw=2)
    # # plt.plot(g_l_diff * 5, lw=2)
    # plt.plot(g_l, lw=2)
    # plt.plot(g_l_new, lw=2)
    # plt.legend(['orig', 'offset', 'new'])

    # plt.figure()
    # plt.plot(g_l_diff, lw=2)
    # plt.show()
    # g_l_new = temp % (2 * math.pi)

    while max(g_l_new) > 2 * math.pi:
        g_l_new = where(g_l_new >= 2 * math.pi - 0.001, g_l_new - 2 * math.pi, g_l_new)
    # plt.figure()
    # plt.plot(_g_l, lw=2)
    # plt.plot(g_l, lw=2)
    # plt.plot(g_l_new, lw=2)
    # plt.plot(temp)
    # plt.legend(['orig', 'offset', 'new'])
    # plt.show()

    return g_l_new

def ReadDataFromText(file_path):
    """load joint angle from text file

    Args:
        file_path (_type_): _description_

    Returns:
        joint angle and time series data
    """
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
    sw_hip = np.array(sw_hip_list)
    st_knee = np.array(st_knee_list)
    sw_knee = np.array(sw_knee_list)
    
    times = np.arange(0, st_hip.shape[0])*0.001
    
    txt_file.close()
    
    return st_hip, sw_hip, st_knee, sw_knee, times

def ReadDataFromCSV(file_path):
    """Read data from MoCop csv file

    Args:
        file_path (_type_): _description_

    Returns:
        _type_: _description_
    """
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

    return right_hip_angle_left_swing,left_hip_angle_left_swing,right_knee_angle_left_swing,left_knee_angle_left_swing,left_swing_time
    
