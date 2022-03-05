import time
from copy import deepcopy
from math import cos, sin

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits import mplot3d
from scipy.interpolate import interp1d

from .converter import cart_joint, d_2_r, joint_cart, r_2_d
from .util import cal_link_angular_moment as CLAM
from .util import cal_link_com, plot_gait_ankle, smooth_data_aver


class bodyM:
    """A class used to simulate lower limbs movement using kinematics from joint angles / cartesian position.
    
    The coordinates system used here is x positive for walking direction, z 
    positive points upward.
    Typical use case:
    1.
        obj = bodyM(body_para, time=time, traj=traj, mode="real")
        obj.plot_traj_car()
        obj.plot_traj_car(if_global=True)
        obj.gait_animation_2D('test.gif')

    2.
        obj = bodyM(body_para, time=time, traj=traj, mode="crafted", 
                        num_steps=1000, dura=1.02)
        obj.plot_traj_car()
        obj.plot_traj_car(if_global=True)
        obj.gait_animation_2D('test.gif')
    """

    def __init__(
        self, body_para, time=None, traj=None, mode=None, debug=False, **kwargs
    ):
        """Initialize the class.
        Args:
            body_para, Body parameters. Refer to util.ideal_body_p.
            time, Time stamps.
            traj, A dictionary of cartesian trajectories, refer to set_traj_cart.
            mode, Available: real (real trajectory); crafted (manually made 
            trajectory). When set to crafeted, trajectory will be automatically 
            generated. Noted, addtionaly parameters need to be passed.
            debug, If set to True, more information will printed or ploted.
            kwargs, Optional parameters to generate the trajectory, including: 
            cycle, if_assign, gait_angle, gait_height. Omit if not use crafted gait.
        """
        if body_para is None:
            raise ValueError("Body parameters are not given")

        self.body_para = body_para
        self.l1 = self.body_para["thign_len"]
        self.l2 = self.body_para["shank_len"]
        self.debug = debug
        self.mode = mode

        if mode == "real":
            if time is None:
                raise ValueError("No time input!")
            self.set_traj_cart(traj, time)
        else:
            self.craft_gait(**kwargs)

        self.X = None
        self.Y = None

    def set_traj_cart(self, traj, time):
        """Set trajectory by ankle cartesian trajecotries.
        
        Joints trajectories and other joints' cartesian trajectories will be 
        automatically generated. Refer to code for deatil. These trajectory 
        should be in local frame.
        """
        self.xf_r = traj["xf_r"]
        self.xf_l = traj["xf_l"]
        self.zf_r = traj["zf_r"]
        self.zf_l = traj["zf_l"]
        self.time = time

        self.gen_joint()
        self.gen_cart()
        self.num_steps = self.time.shape[0]

    def set_traj_global(self, traj, time):
        """Manually set the global trajecotries.
        Refer to code for details.
        """
        self.g_xf_r = traj["xf_r"]
        self.g_xf_l = traj["xf_l"]
        self.g_zf_r = traj["zf_r"]
        self.g_zf_l = traj["zf_l"]

        self.g_xk_r = traj["xk_r"]
        self.g_xk_l = traj["xk_l"]
        self.g_zk_r = traj["zk_r"]
        self.g_zk_l = traj["zk_l"]

        self.g_xh_r = traj["xh_r"]
        self.g_xh_l = traj["xh_l"]
        self.g_zh_r = traj["zh_r"]
        self.g_zh_l = traj["zh_l"]

    def set_hs_to(self, val):
        """Set heal-strike and toe-off event.
        Refer to code for details.
        """
        self.r_hs = val["r_hs"]
        self.l_hs = val["l_hs"]
        self.r_to = val["r_to"]
        self.l_to = val["l_to"]

    def get_traj(self):
        """Get ankel trajectory."""
        return self.xf_r, self.zf_r, self.xf_l, self.zf_l

    def get_traj_global(self):
        """Get ankel global trajectory."""
        return self.g_xf_r, self.g_zf_r, self.g_xf_l, self.g_zf_l

    def gen_cart(self):
        """Generate cartesian trajectory w.r.t trajectories in joint space."""
        assert hasattr(self, "theta_h_r")
        (
            self.xh_r,
            self.zh_r,
            self.xk_r,
            self.zk_r,
            self.xf_r,
            self.zf_r,
        ) = joint_cart(self.theta_h_r, self.theta_k_r, self.l1, self.l2)
        (
            self.xh_l,
            self.zh_l,
            self.xk_l,
            self.zk_l,
            self.xf_l,
            self.zf_l,
        ) = joint_cart(self.theta_h_l, self.theta_k_l, self.l1, self.l2)

    def get_time(self):
        """Get timestamps."""
        return self.time

    def get_joint(self):
        """Get trajectories in joint space."""
        return self.theta_h_r, self.theta_k_r, self.theta_h_l, self.theta_k_l

    def gen_joint(self):
        """Generate trajectories in joint space."""
        self.theta_h_r, self.theta_k_r = cart_joint(
            self.xf_r, self.zf_r, self.l1, self.l2
        )
        self.theta_h_l, self.theta_k_l = cart_joint(
            self.xf_l, self.zf_l, self.l1, self.l2
        )

    def craft_gait_i(
        self, order="+", cycle=1, gait_angle=120, gait_height=0.6, **kwargs
    ):
        """Internal function used to generate a gait trajectory.
        Args:
            order, Specify the initail status (right or left).
            cycle, How many gait cycles should be generated.
            gait_angle, Determined the step length.
            gait_heigth, Determined ground's position.

        Returns:
            Array of ankle's x trajectory.
            Array of ankle's z trajectory.
        """
        theta = d_2_r(gait_angle / 2)
        d = (self.l1 + self.l2) * sin(theta)
        dval = 2 * d / self.num_steps

        ground = -cos(theta) * (self.l1 + self.l2) + 0.01
        self.ground = ground

        time_steps_half = int(self.num_steps / 2)

        c = gait_height
        a = -c / (d ** 2)

        if order == "+":
            xp = np.linspace(-d, d - dval, num=time_steps_half)
            xp = np.append(xp, np.linspace(d, -d + dval, num=time_steps_half))
            yp = a * np.power(xp[0:time_steps_half], 2) + c
            yp = np.append(yp, np.zeros((1, time_steps_half)))
        elif order == "-":
            xp = np.linspace(d, -d + dval, num=time_steps_half)
            xp = np.append(xp, np.linspace(-d, d - dval, num=time_steps_half))
            yp = a * np.power(xp[0:time_steps_half], 2) + c
            yp = np.append(np.zeros((1, time_steps_half)), yp)

        temp_y = yp
        temp_x = xp
        for _ in range(cycle - 1):
            yp = np.append(yp, temp_y)
            xp = np.append(xp, temp_x)

        yp += ground

        xp = smooth_data_aver(20, xp)
        yp = smooth_data_aver(20, yp)

        if self.debug:
            plt.plot(xp, "b", lw=2)
            plt.plot(yp, "c", lw=2)
            plt.legend(["xp", "yp"])
            plt.show()

        return xp, yp

    def craft_gait(self, cycle=1, if_assign=True, **kwargs):
        """Generate a gait trajectory in Cartesian coordinates.
        cycle, How many full gait should be generated.
        if_assign, If assign value to this bodyM object.
        """
        self.num_steps = kwargs["time_steps"]
        dura = kwargs["duration"]
        self.time = np.linspace(0, dura, self.num_steps)

        self.xf_r, self.zf_r = self.craft_gait_i(cycle=cycle, **kwargs)
        self.xf_l, self.zf_l = self.craft_gait_i(order="-", cycle=cycle, **kwargs)

        theta_h_r, theta_k_r = cart_joint(self.xf_r, self.zf_r, self.l1, self.l2)
        theta_h_l, theta_k_l = cart_joint(self.xf_l, self.zf_l, self.l1, self.l2)

        if if_assign:
            self.theta_h_r = theta_h_r
            self.theta_k_r = theta_k_r
            self.theta_h_l = theta_h_l
            self.theta_k_l = theta_k_l

        self.gen_cart()

        return theta_h_r, theta_k_r, theta_h_l, theta_k_l

    def float_hip_x(self):
        """Floating trajectory in x direction. 

        Not fixed frame to hip in x direction.
        """
        self.copy_float()
        self.float_hip_x_i()

    def copy_float(self):
        """Internal function."""
        self.g_xf_l = deepcopy(self.xf_l)
        self.g_xk_l = deepcopy(self.xk_l)
        self.g_xh_l = deepcopy(self.xh_l)
        self.g_zf_l = deepcopy(self.zf_l)
        self.g_zk_l = deepcopy(self.zk_l)
        self.g_zh_l = deepcopy(self.zh_l)

        self.g_xf_r = deepcopy(self.xf_r)
        self.g_xk_r = deepcopy(self.xk_r)
        self.g_xh_r = deepcopy(self.xh_r)
        self.g_zf_r = deepcopy(self.zf_r)
        self.g_zk_r = deepcopy(self.zk_r)
        self.g_zh_r = deepcopy(self.zh_r)

    def float_hip_x_i(self):
        """Internal function to float the frame.
        Vital leg is determined by the position relative to ground.
        """
        diff_accu = 0
        for i in range(self.num_steps):
            if (
                self.zf_l[i] <= self.ground + 0.03
                and i != 0
                and (
                    self.xf_l[i - 1] - self.xf_l[i] > 0
                    or self.zf_r[i] > self.ground + 0.01
                )
            ):
                diff = self.xf_l[i - 1] - self.xf_l[i]
                diff_accu += diff
                self.g_xf_l[i] += diff_accu
                self.g_xk_l[i] += diff_accu
                self.g_xh_l[i] += diff_accu
                self.g_xh_r[i] += diff_accu
                self.g_xf_r[i] += diff_accu
                self.g_xk_r[i] += diff_accu
            elif (
                self.zf_r[i] <= self.ground + 0.03
                and i != 0
                and (
                    self.xf_r[i - 1] - self.xf_r[i] > 0
                    or self.zf_l[i] > self.ground + 0.01
                )
            ):
                diff = self.xf_r[i - 1] - self.xf_r[i]
                diff_accu += diff
                self.g_xf_r[i] += diff_accu
                self.g_xk_r[i] += diff_accu
                self.g_xh_r[i] += diff_accu
                self.g_xf_l[i] += diff_accu
                self.g_xk_l[i] += diff_accu
                self.g_xh_l[i] += diff_accu
            elif i > 0:
                raise ValueError("Wrong!")

    def plot_traj_car(self, if_global=False):
        """Plot trajectories in cartesian space."""
        if if_global:
            plot_gait_ankle(self.g_xf_r, self.g_xf_l, self.g_zf_r, self.g_zf_l)
        else:
            plot_gait_ankle(self.xf_r, self.xf_l, self.zf_r, self.zf_l)

    def sep_front_craft(self, tol=0.05):
        """Seperate which leg is the vital leg.
        
        For crafted trajectory. Useful in calculating ZMP and XCoM.
        """
        assert hasattr(self, "g_zf_r"), "Have to float first!"

        index_l = np.less(self.g_zf_l, self.ground + tol)
        index_r = np.less(self.g_zf_r, self.ground + tol)

        index_com = index_l == index_r

        index_l[index_com] = False
        index_r[index_com] = False
        index_com = np.nonzero(index_com)[0]

        return index_l, index_r, index_com

    def sep_front_real(self):
        """Seperate which leg is the vital leg.
        
        For real trajectory. Needs hs and to labeling. Useful in calculating ZMP 
        and XCoM.
        """
        index_r = self.sep_front_real_i(self.r_hs, self.r_to)
        index_l = self.sep_front_real_i(self.l_hs, self.l_to)

        index_com = index_l == index_r
        index_l[index_com] = False
        index_r[index_com] = False

        return index_l, index_r, np.nonzero(index_com)[0]

    def sep_front_real_i(self, hs, to):
        """Internal function.
        hs, heal strike.
        to, toe-off
        """
        index = np.zeros(len(self.g_xf_r), dtype=bool)
        i_hs = np.where(hs == True)[0]
        i_to = np.where(to == True)[0]

        i_buffer = []
        for i in i_hs:
            temp = np.where(i_to > i)[0]
            if temp.size > 0:
                to = i_to[temp[0]]
                index[i:to] = True
                i_buffer.append(to)
            else:
                index[i:] = True

        for i in i_to:
            if i in i_buffer:
                continue
            index[:i] = True

        return index

    def cal_XCoM(self, plot=False):
        """Calculate the XCoM"""
        x_com = self.g_xh_l
        z_com = (
            self.g_zh_l + self.body_para["COM_height"] - self.body_para["hip_height"]
        )

        vx = np.gradient(x_com, self.time)

        if self.mode == "craft":
            index_l, index_r, index_com = self.sep_front_craft()
        else:
            index_l, index_r, index_com = self.sep_front_real()

        length_l = np.power(self.g_xf_l - x_com, 2) + np.power(self.g_zf_l - z_com, 2)
        length_l = np.sqrt(length_l)
        length_r = np.power(self.g_xf_r - x_com, 2) + np.power(self.g_zf_r - z_com, 2)
        length_r = np.sqrt(length_r)

        length = np.zeros((self.num_steps))

        for i in index_com.tolist():
            if self.g_xf_l[i] >= self.g_xf_r[i]:
                index_l[i] = True
            else:
                index_r[i] = True

        length[index_l] = length_l[index_l]
        length[index_r] = length_r[index_r]

        freq0 = np.sqrt(9.8 / length)

        self.xcom = x_com + vx / freq0

        if plot:
            _, ax1 = plt.subplots()
            line1 = ax1.plot(self.xcom, "b", lw=4)
            ax1.set_xlabel("samples")
            ax1.set_ylabel("displacement", color="b")

            ax2 = ax1.twinx()
            ax2.set_ylabel("displacement", color="c")
            line2 = ax2.plot(self.xcom - x_com, "c", lw=2)

            lines = line1 + line2
            ax1.legend(lines, ["xcom", "xcom - x_com"])

        return self.xcom, self.xcom - x_com

    def get_link_dyn(self):
        """Calculate dynamic properties for each link."""
        x1_list = [
            self.g_xf_l,
            self.g_xk_l,
            self.g_xf_r,
            self.g_xk_r,
            self.g_zf_l,
            self.g_zk_l,
            self.g_zf_r,
            self.g_zk_r,
        ]

        x2_list = [
            self.g_xk_l,
            self.g_xh_l,
            self.g_xk_r,
            self.g_xh_r,
            self.g_zk_l,
            self.g_zh_l,
            self.g_zk_r,
            self.g_zh_r,
        ]

        shank_c = 1 - self.body_para["shank_COM_coe"]
        thign_c = 1 - self.body_para["thign_COM_coe"]

        c_list = [
            shank_c,
            thign_c,
            shank_c,
            thign_c,
            shank_c,
            thign_c,
            shank_c,
            thign_c,
        ]

        if self.debug:
            titles = [
                "CoM left shank x",
                "CoM left thign x",
                "CoM right shank x",
                "CoM right thign x",
                "CoM left shank z",
                "CoM left thign z",
                "CoM right shank z",
                "CoM right thign z",
            ]

            x_labels = ["Samples" for i in range(8)]

            y_labels = ["x diplacement mm" for i in range(4)]
            y_labels.extend(["z diplacement mm" for i in range(4)])

            legends = [
                ["ankle x", "shank com x", "knee x"],
                ["knee x", "thign com x", "hip x"],
                ["ankle x", "shank com x", "knee x"],
                ["knee x", "thign com x", "hip x"],
                ["ankle z", "shank com z", "knee z"],
                ["knee z", "thign com z", "hip z"],
                ["ankle z", "shank com z", "knee z"],
                ["knee z", "thign com z", "hip z"],
            ]

            coms = cal_link_com(
                x1_list,
                x2_list,
                c_list,
                plot=True,
                titles=titles,
                x_labels=x_labels,
                y_labels=y_labels,
                legends=legends,
            )
        else:
            coms = cal_link_com(x1_list, x2_list, c_list, plot=False)

        dcoms = []
        ddcoms = []
        for i, _ in enumerate(coms):
            dcoms.append(np.gradient(coms[i], 0.001))
            ddcoms.append(np.gradient(dcoms[i], 0.001))

        return coms, dcoms, ddcoms

    def cal_ZMP(self, plot=False):
        """Calculate ZMP"""
        w = self.body_para["hip_width"] / 2

        L_dot = []
        X = []
        Y = []
        Z = []
        ddX = []
        ddY = []
        ddZ = []
        M = [
            self.body_para["shank_weight"],
            self.body_para["thign_weight"],
            self.body_para["shank_weight"],
            self.body_para["thign_weight"],
            self.body_para["weight"]
            - self.body_para["thign_weight"]
            - self.body_para["shank_weight"],
        ]

        coms, dcoms, ddcoms = self.get_link_dyn()
        (
            shank_l_x,
            thign_l_x,
            shank_r_x,
            thign_r_x,
            shank_l_z,
            thign_l_z,
            shank_r_z,
            thign_r_z,
        ) = coms

        (
            shank_l_dx,
            thign_l_dx,
            shank_r_dx,
            thign_r_dx,
            shank_l_dz,
            thign_l_dz,
            shank_r_dz,
            thign_r_dz,
        ) = dcoms

        (
            shank_l_ddx,
            thign_l_ddx,
            shank_r_ddx,
            thign_r_ddx,
            shank_l_ddz,
            thign_l_ddz,
            shank_r_ddz,
            thign_r_ddz,
        ) = ddcoms

        shank_l_y = np.ones(self.num_steps) * w
        shank_r_y = np.ones(self.num_steps) * (-w)
        thign_l_y = np.ones(self.num_steps) * w
        thign_r_y = np.ones(self.num_steps) * (-w)

        shank_l_dy = np.zeros((self.num_steps))
        shank_r_dy = np.zeros((self.num_steps))
        thign_l_dy = np.zeros((self.num_steps))
        thign_r_dy = np.zeros((self.num_steps))

        shank_l_ddy = np.zeros((self.num_steps))
        shank_r_ddy = np.zeros((self.num_steps))
        thign_l_ddy = np.zeros((self.num_steps))
        thign_r_ddy = np.zeros((self.num_steps))

        _, L_dot_shank_l = CLAM(
            shank_l_x, shank_l_y, shank_l_z, shank_l_dx, shank_l_dy, shank_l_dz, M[0]
        )
        _, L_dot_thign_l = CLAM(
            thign_l_x, thign_l_y, thign_l_z, thign_l_dx, thign_l_dy, thign_l_dz, M[1]
        )
        _, L_dot_shank_r = CLAM(
            shank_r_x, shank_r_y, shank_r_z, shank_r_dx, shank_r_dy, shank_r_dz, M[2]
        )
        _, L_dot_thign_r = CLAM(
            thign_r_x, thign_r_y, thign_r_z, thign_r_dx, thign_r_dy, thign_r_dz, M[3]
        )

        g_ddxh_l = np.gradient(np.gradient(self.g_xh_l, self.time), self.time)
        g_ddzh_l = np.gradient(np.gradient(self.g_zh_l, self.time), self.time)

        A_x = np.zeros((self.num_steps))
        A_y = np.zeros((self.num_steps))
        C = np.zeros((self.num_steps))

        for i in range(self.num_steps):
            X.append(
                [shank_l_x[i], thign_l_x[i], shank_r_x[i], thign_r_x[i], self.g_xh_l[i]]
            )
            Y.append([shank_l_y[i], thign_l_y[i], shank_r_y[i], thign_r_y[i], 0])
            Z.append(
                [shank_l_z[i], thign_l_z[i], shank_r_z[i], thign_r_z[i], self.g_zh_l[i]]
            )

            ddX.append(
                [
                    shank_l_ddx[i],
                    thign_l_ddx[i],
                    shank_r_ddx[i],
                    thign_r_ddx[i],
                    g_ddxh_l[i],
                ]
            )
            ddY.append(
                [shank_l_ddy[i], thign_l_ddy[i], shank_r_ddy[i], thign_r_ddy[i], 0]
            )
            ddZ.append(
                [
                    shank_l_ddz[i],
                    thign_l_ddz[i],
                    shank_r_ddz[i],
                    thign_r_ddz[i],
                    g_ddzh_l[i],
                ]
            )

            L_dot.append(
                [
                    L_dot_shank_l[i],
                    L_dot_thign_l[i],
                    L_dot_shank_r[i],
                    L_dot_thign_r[i],
                    [0, 0, 0],
                ]
            )

        for i in range(self.num_steps):
            for j, m in enumerate(M):
                A_x[i] += (
                    m * (Y[i][j] + 9.8) * X[i][j]
                    - m * ddX[i][j] * Y[i][j]
                    - L_dot[i][j][2]
                )
                A_y[i] += (
                    m * (Y[i][j] + 9.8) * Z[i][j]
                    - m * ddZ[i][j] * Y[i][j]
                    - L_dot[i][j][0]
                )
                C[i] += m * (Y[i][j] + 9.8)

        x_zmp = A_x / C - self.g_xh_l
        y_zmp = A_y / C

        if plot:
            _ = plt.figure()
            plt.plot(x_zmp, lw=2)
            plt.xlabel("samples")
            plt.ylabel("displacemen")
            plt.title("zmp in x direction")
            _ = plt.figure()
            plt.plot(y_zmp, lw=2)
            plt.xlabel("samples")
            plt.ylabel("displacemen")
            plt.title("zmp in y direction")

        return x_zmp, y_zmp

    def animation_data_2D(self, if_global):
        """Generate Animation data."""
        if if_global:
            length = len(self.xh_r)
            self.X = [
                [
                    self.g_xf_r[i * 10],
                    self.g_xk_r[i * 10],
                    self.g_xh_r[i * 10],
                    self.g_xh_l[i * 10],
                    self.g_xk_l[i * 10],
                    self.g_xf_l[i * 10],
                ]
                for i in range(int(length / 10))
            ]
            self.Y = [
                [
                    self.g_zf_r[i * 10],
                    self.g_zk_r[i * 10],
                    self.g_zh_r[i * 10],
                    self.g_zh_l[i * 10],
                    self.g_zk_l[i * 10],
                    self.g_zf_l[i * 10],
                ]
                for i in range(int(length / 10))
            ]
            highest = np.max([np.max(self.g_zf_r), np.max(self.g_zf_l)])
            lowest = np.min([np.min(self.g_zf_r), np.min(self.g_zf_l)])
            farest = np.max([np.max(self.g_xf_r), np.max(self.g_xf_l)])
            nearest = np.min([np.min(self.g_xf_r), np.min(self.g_xf_l)])
            return highest, lowest, farest, nearest
            # self.X = [[self.g_xf_r[i], self.g_xk_r[i], self.g_xh_r[i], \
            # 			self.g_xh_l[i], self.g_xk_l[i], self.g_xf_l[i]]	\
            # 			for i, _ in enumerate(self.xh_r)]
            # self.Y = [[self.g_zf_r[i], self.g_zk_r[i], self.g_zh_r[i], \
            # 			self.g_zh_l[i], self.g_zk_l[i], self.g_zf_l[i]]	\
            # 			for i, _ in enumerate(self.zh_r)]
        else:
            self.X = [
                [
                    self.xf_r[i],
                    self.xk_r[i],
                    self.xh_r[i],
                    self.xh_l[i],
                    self.xk_l[i],
                    self.xf_l[i],
                ]
                for i, _ in enumerate(self.xh_r)
            ]
            self.Y = [
                [
                    self.zf_r[i],
                    self.zk_r[i],
                    self.zh_r[i],
                    self.zh_l[i],
                    self.zk_l[i],
                    self.zf_l[i],
                ]
                for i, _ in enumerate(self.zh_r)
            ]

    def gait_animation_2D(self, time=None, interval=0, if_global=False, save=None):
        """Generate 2D animation."""
        if self.X is None or self.Y is None:
            highest, lowest, farest, nearest = self.animation_data_2D(if_global)

        def init():
            point_ani.set_data(self.X[0], self.Y[0])
            plt.axis([-1, 3, -2, 2])
            return (point_ani,)

        def update_points(i):
            point_ani.set_data(self.X[i], self.Y[i])
            plt.axis([-1, 3, -2, 2])
            return (point_ani,)

        num_frame = len(self.X)
        fig = plt.figure(tight_layout=True)
        plt.xlabel("Y")
        plt.ylabel("Z")
        plt.plot([nearest, nearest], [-2, 2], lw=2, color="b")
        plt.plot([-1, 3], [highest, highest], lw=2, color="r")
        plt.legend(["ylimit", "zlimit"])
        plt.plot([-1, 3], [lowest, lowest], lw=2, color="r")
        plt.plot([farest, farest], [-2, 2], lw=2, color="b")
        (point_ani,) = plt.plot(self.X[0], self.Y[0], lw=2)
        self.ani = animation.FuncAnimation(
            fig,
            update_points,
            np.arange(0, num_frame),
            init_func=init,
            interval=interval,
            blit=True,
        )

        if save is not None:
            self.ani.save(save, writer="imagemagick", fps=100)

        return self.ani
