import math
import sys
from copy import deepcopy

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
from utils import save_csv


class cs:
    """
    short for canonical system. A internal class.
    """

    def __init__(
        self, dt: float, ax=1.0, run_time_coe=1.0, pattern="discrete", **kwargs
    ):
        """Default values from Schaal (2012)

        ax float: a gain term on the dynamical system
        pattern string: either 'discrete' or 'rhythmic'
        """
        self.run_time = 1.0 * run_time_coe

        self.ax = ax
        self.dt = dt
        self.timesteps = int(round(self.run_time / self.dt))
        self.reset_state()

    def rollout(self, tau=1.0, **kwargs):
        """Generate x for open loop movements.
        """
        timesteps = int(self.timesteps / tau)
        self.x_track = np.zeros(timesteps)

        self.reset_state()
        for t in range(timesteps):
            self.x_track[t] = self.x
            self.step(**kwargs)

        return self.x_track

    def reset_state(self):
        # Reset the system state
        self.x = 1.0

    def step(self, tau=1.0, error_coupling=1.0, **kwargs):
        """Generate a single step of x for discrete
        (potentially closed) loop movements.
        Decaying from 1 to 0 according to dx = -ax*x.

        tau float: gain on execution time
                                increase tau to make the system execute faster
        error_coupling float: slow down if the error is > 1 (not used as no coupling)
        """
        self.x += (-self.ax * self.x * error_coupling) * tau * self.dt
        return self.x


class DMP:
    """
    Discrete DMP.
    Implementation of Dynamic Motor Primitives,
    as described in Dr. Stefan Schaal's (2002) paper.
    """

    def __init__(self, path, time, ay=None, by=None, dt: float = 0.001, n_bfs=500, isz=False):
        """
        path, Ideal trajectory, n_dmps X T
        time, T
        ay, coefficient for DMP, n_dmps
        by, same to ay
        dt, time interval
        n_bfs, number of basis functions.
        isz, if the path is for z dirction. (fix offset for in z)
        """

        def copy_reshape(_x):
            x = deepcopy(_x)
            return x.reshape(1, x.shape[0]) if x.ndim == 1 else x

        self.n_dmps, self.dt, self.n_bfs, self.isz = (
            path.shape[0],
            dt,
            n_bfs,
            isz,
        )

        path = copy_reshape(path)
        self.time = copy_reshape(time)

        self.path_bound = np.zeros((path.shape[0], 2))
        for i in range(path.shape[0]):
            self.path_bound[i, 0] = path[i, 0]
            self.path_bound[i, 1] = path[i, -1]

        
        # self.offset = path[:, 0]; 
        # self.path = path - self.offset.reshape(self.n_dmps, -1)
        self.offset = np.zeros(self.n_dmps)
        self.path = path
        self.goal = np.copy(self.path[:, -1])
        self.y0 = np.zeros(self.n_dmps)
        self.y0 = np.copy(self.path[:, 0])

        self.ay = np.ones(self.n_dmps) * 50.0 if ay is None else ay  # Schaal 2012
        self.by = self.ay / 4.0 if by is None else by  # Schaal 2012
        assert self.ay.size == self.path.shape[0] == self.by.size

        # set up the CS
        self.cs = cs(dt=dt, run_time_coe=self.time[0, -1])
        self.timesteps = int(round(self.cs.run_time / self.dt))

        self.reset_state()
        self.check_offset()
        self.gen_centers()
        self.h = np.ones(n_bfs) * n_bfs ** 1.5# / self.c / self.cs.ax
        # print(self.c, self.h)

    def check_offset(self):
        """Check to see if initial position and goal are the same
        if they are, offset slightly so that the forcing term is not 0"""
        for d in range(self.n_dmps):
            if abs(self.y0[d] - self.goal[d]) < 1e-4:
                self.goal[d] += 0.1

    def gen_centers(self):
        """Set the centre of the Gaussian basis
        functions be spaced evenly throughout run time"""
        # desired activations throughout time
        des_c = np.linspace(0, self.cs.run_time, self.n_bfs)
        
        self.c = np.ones(len(des_c))
        for n in range(len(des_c)):
            # finding x for desired times t
            self.c[n] = np.exp(-self.cs.ax * des_c[n])

    def genVarianceSig(self, c_t, sig_coe=1.0):
        """No use."""
        h0 = np.ones(self.n_bfs)
        for i, c in enumerate(self.c):
            if i == 0:
                sigma = (c - self.c[i + 1]) * sig_coe
            if i == self.n_bfs - 1:
                sigma = (self.c[i - 1] - c) * sig_coe
            else:
                sigma = (c - self.c[i + 1]) * sig_coe

            h0[i] = 1 / (sigma * math.sqrt(2))

        self.h = h0 * sig_coe

        return self.h

    def genCenterGrad(self, path, time, dens):
        """No use. Assign kernel center w.r.t gradient of trajectories."""
        path_grad = np.abs(np.gradient(path, time))
        path_grad_csum = np.cumsum(path_grad)
        f = scipy.interpolate.interp1d(path_grad_csum, time)

        c_t = [0]
        temp = f(1 / dens)
        i = 1
        while not math.isnan(temp):
            c_t.append(temp.item())
            i += 1
            if 1 / dens * i >= path_grad_csum[-1]:
                break
            temp = f(1 / dens * i)

        c_t.append(time[-1])

        self.c_t = np.array(c_t)
        self.c = np.exp(-self.cs.ax * self.c_t)

        return self.c, self.c_t

    def gen_front_term(self, x, dmp_num):
        return x
        # return x * (self.goal[dmp_num] - self.y0[dmp_num])

    def gen_psi(self, x):
        """Kernel activations."""
        if isinstance(x, np.ndarray):
            x = x[:, None]

        # import matplotlib.pyplot as plt

        # f,axs = plt.subplots(2)
        # axs[0].plot(self.h)
        # axs[1].plot(self.c)
        # plt.show()

        return np.exp(-self.h * (x - self.c) ** 2)
        

    def gen_weights(self, f_target):
        """Generate a set of weights over the basis functions such
        that the target forcing term trajectory is matched.

        f_target np.array: the desired forcing term trajectory
        """

        # calculate x and psi
        x_track = self.cs.rollout()
        self.cs.step()  # to the end point
        x_track = np.append(x_track, self.cs.x)
        psi_track = self.gen_psi(x_track)

        # efficiently calculate BF weights using weighted linear regression
        self.w = np.zeros((self.n_dmps, self.n_bfs))
        for d in range(self.n_dmps):
            # spatial scaling term, set the scaling as 1.0 rahter self.goal[d] - self.y0[d]
            k = 1.0
            for b in range(self.n_bfs):
                numer = np.sum(x_track * psi_track[:, b] * f_target[:, d])
                denom = np.sum(x_track ** 2 * psi_track[:, b])
                self.w[d, b] = numer / denom
                if abs(k) > 1e-5:
                    self.w[d, b] /= k

        self.w = np.nan_to_num(self.w)

    def imitate_path(self, plot=False):
        """Takes in a desired trajectory and generates the set of
        system parameters that best realize this path.

        y_des list/array: the desired trajectories of each DMP
                                            should be shaped [n_dmps, run_time]
        """

        # set initial state and goal
        time = np.arange(0, self.time[0, -1] + self.dt, self.dt)
        eps = 1e-10
        interpolate_time = self.time.ravel()
        interpolate_time[0] -= eps  # Ensure boundary is not exceeded.
        interpolate_time[-1] += eps

        # generate function to interpolate the desired trajectory
        # +1 to include the starting point.
        path = np.zeros((self.n_dmps, self.timesteps + 1))
        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(interpolate_time, self.path[d, :])
            for t in range(self.timesteps + 1):  # +1 to include the starting point.
                path[d, t] = path_gen(time[t])
        y_des = path

        dy_des = np.gradient(y_des, self.dt, axis=1)
        ddy_des = np.gradient(dy_des, self.dt, axis=1)

        f_target = np.zeros((y_des.shape[1], self.n_dmps))
        # find the force required to move along this trajectory
        for d in range(self.n_dmps):
            f_target[:, d] = ddy_des[d] - self.ay[d] * (
                self.by[d] * (self.goal[d] - y_des[d]) - dy_des[d]
            )

        #'''test target differential'''
        # y_temp = np.zeros_like(y_des)
        # dy_temp = np.zeros_like(dy_des)
        # ddy_temp = np.zeros_like(ddy_des)
        # y_temp[:, 0] = y_des[:, 0]
        # dy_temp[:, 0] = dy_des[:, 0]
        # for d in range(self.n_dmps):
        #     for i in range(1, y_des.shape[1]):
        #         ddy_temp[d, i-1] = self.ay[d] * (self.by[d] * (self.goal[d] - y_temp[d, i-1]) - dy_temp[d, i-1]) + f_target[i, d]
        #         dy_temp[d, i] = dy_temp[d, i-1] + ddy_temp[d, i-1] * self.dt
        #         y_temp[d, i] = y_temp[d, i-1] + dy_temp[d, i-1] * self.dt

        # import matplotlib.pyplot as plt
        # f = plt.figure()
        # for d in range(self.n_dmps): 
        #     plt.plot(y_des[d], label="ori")
        #     plt.plot(y_temp[d], label="temp")
        # plt.legend()
        # plt.show()
        # sys.exit()

        # efficiently generate weights to realize f_target
        self.gen_weights(f_target)

        if plot is True:
            # plot the basis function activations
            import matplotlib.pyplot as plt

            plt.figure()
            plt.subplot(211)
            psi_track = self.gen_psi(self.cs.rollout())
            plt.plot(psi_track)
            plt.title("basis functions")

            # plot the desired forcing function vs approx
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                plt.plot(f_target[:, ii], "--", label="f_target %i" % ii)
            for ii in range(self.n_dmps):
                plt.subplot(2, self.n_dmps, self.n_dmps + 1 + ii)
                print("w shape: ", self.w.shape)
                plt.plot(
                    np.sum(psi_track * self.w[ii], axis=1) * self.dt,
                    label="w*psi %i" % ii,
                )
                plt.legend()
            plt.title("DMP forcing function")
            plt.tight_layout()
            plt.show()

        self.reset_state()
        return y_des, time

    def rollout(self, goal=1.0, tau=1.0, timesteps=None, reset=True, **kwargs):
        """Generate a system trial, no feedback is incorporated."""
        if reset == True:
            self.reset_state()

        if timesteps is None:
            timesteps = int(self.timesteps / tau)

        rollout_time = np.linspace(0, timesteps * self.dt, timesteps+1)

        goal_temp = deepcopy(self.goal)
        self.goal = goal * self.goal
        offset = self.offset if self.isz else goal * self.offset

        # set up tracking vectors
        y_track = np.zeros((timesteps+1, self.n_dmps))
        dy_track = np.zeros((timesteps, self.n_dmps))
        ddy_track = np.zeros((timesteps, self.n_dmps))
        
        y_track[0] = self.y
        for t in range(timesteps):
            # run and record timestep
            y_track[t+1], dy_track[t], ddy_track[t] = self.step(goal, tau, **kwargs)

        self.goal = goal_temp

        return y_track + offset, dy_track, ddy_track, rollout_time

    def reset_state(self):
        """Reset the system state"""
        self.y = self.y0.copy()
        self.dy = np.zeros(self.n_dmps)
        self.ddy = np.zeros(self.n_dmps)
        self.cs.reset_state()

    def step(self, tau=1.0, error=0.0, external_force=None, **kwargs):
        """Run the DMP system for a single timestep.

        tau float: scales the timestep
                                increase tau to make the system execute faster
        error float: optional system feedback
        """

        # error_coupling = 1.0 / (1.0 + error)
        error_coupling = 1.0
        # run canonical system
        x = self.cs.step(tau=tau, error_coupling=error_coupling)

        # generate basis function activation
        psi = self.gen_psi(x)

        for d in range(self.n_dmps):
            # generate the forcing term
            f = self.gen_front_term(x, d) * (np.dot(psi, self.w[d])) / np.sum(psi)

            # DMP acceleration
            self.ddy[d] = (
                self.ay[d] * (self.by[d] * (self.goal[d] - self.y[d]) - self.dy[d]) + f
            )
            if external_force is not None:
                self.ddy[d] += external_force[d]
            self.dy[d] += self.ddy[d] * tau * self.dt * error_coupling
            self.y[d] += self.dy[d] * tau * self.dt * error_coupling

        return self.y, self.dy, self.ddy

    def step_real(self, gait_phase, y_, dy_, scale=1.0, goal_offset=0.0, tau=1.0, extra_force=None):
        """Run one step for real cases.

            Args:
                gait_phase: current gait_phase in %.
                y_, CUrrent position in array.
                dy_, CUrrent velocity in array.
                scale, scaling arong goal position
                goal_offset: goal offset
                tau float: scales the timestep of calculation
                extra_force: optional extral force applied to adjust dmp behavior.
        """ 
        goal_temp = deepcopy(self.goal) # temp save goal value
        self.goal += goal_offset

        # run canonical system
        t = (self.time[-1][-1] * gait_phase + self.dt)  # next step
        x = math.exp(-self.cs.ax * t)

        y = np.zeros(self.n_dmps)
        dy = np.zeros(self.n_dmps)
        ddy = np.zeros(self.n_dmps)

        # generate basis function activation
        psi = self.gen_psi(x)
        psi_sum = np.sum(psi)
        
        # if no provided exteral force 
        if extra_force is None:
            extra_force = np.zeros(self.n_dmps)
        
        for d in range(self.n_dmps):
            # generate the forcing term
            f = self.gen_front_term(x, d) * (np.dot(psi, self.w[d])) / psi_sum * scale[d] + extra_force[d]

            # DMP acceleration
            ddy[d] = self.ay[d] * (self.by[d] * (self.goal[d] - y_[d]) - dy_[d]) + f
            dy[d] = dy_[d] + ddy[d] * tau * self.dt
            y[d] = y_[d] + dy[d] * tau * self.dt

        self.goal = deepcopy(goal_temp) # recover goal value

        return y, dy, ddy

    def full_generation(self, num_steps,y0,new_scale=None,goal_offset=None,forces=None):
        """Run DMP system to generate the entire trajectories

        Args:
            num_steps (_type_): sample data number
            y0 (_type_): initial position
            new_scale (_type_, optional): scaling for DMP. Defaults to None.
            goal_offset (_type_, optional): goal offset. Defaults to None.
            forces (_type_, optional): extra force terms. Defaults to None.

        Returns:
            _type_: generated trajectories
        """
        track = np.zeros((self.n_dmps, num_steps))
        tau = (self.timesteps+1) / num_steps
        track_time = np.arange(num_steps) * self.dt * tau
        # the goal_offset, scale, and initial position for the generated trajectory
        if new_scale is None:
            new_scale = np.ones(self.n_dmps)
        if goal_offset is None:
            goal_offset = np.zeros(self.n_dmps)
        if forces is None:
            forces = np.zeros((self.n_dmps, num_steps))
            
        y = y0
        dy = np.zeros(self.n_dmps)
        for i in range(num_steps):            
            gait_phase = float(i) / num_steps
            y, dy, ddy = self.step_real(gait_phase,y,dy,scale=new_scale,goal_offset=goal_offset,tau=tau,extra_force=forces[:,i])
            track[:, i] = deepcopy(y)
        
        return track, track_time
        
    @property
    def psi_track(self):
        """Generate kernel activations along time."""
        x_track = self.cs.rollout()
        self.cs.step()  # To include starting point.
        x_track = np.append(x_track, self.cs.x)
        psi_track = self.gen_psi(x_track)

        return psi_track

    def save_psi(self, path):
        """Save kernel activations."""
        psi_track = self.psi_track.tolist()
        # +1 to include starting point
        save_csv(path, (self.timesteps + 1, self.n_bfs), psi_track)

    @property
    def para(self):
        """Get DMP paramters."""
        ax = np.ones(self.n_dmps) * self.cs.ax
        dt = np.ones(self.n_dmps) * self.dt
        run_time = np.ones(self.n_dmps) * self.time[-1][-1]
        num_steps = np.ones(self.n_dmps) * self.timesteps
        para_dmp = {
            "a": self.ay,
            "b": self.by,
            "y0": self.y0,
            "g": self.goal,
            "ax": ax,
            "dt": dt,
            "num_steps": num_steps,
            "offset": self.offset,
            "run_time": run_time,
        }
        para_kernel = {"c": self.c, "h": self.h, "w": self.w}

        para = {"dmp": para_dmp, "kernel": para_kernel}

        return para

    def save_para(self, path):
        """Save dmp parameters."""
        dmp_file = path + "_para.csv"
        kernel_file = path + "_kernel.csv"

        row_col_dmp = (9, 4)
        row_col_kernel = (6, self.n_bfs)

        para_ = self.para

        para = para_["dmp"]
        data_list = [
            para["a"],
            para["b"],
            para["y0"],
            para["g"],
            para["ax"],
            para["dt"],
            para["num_steps"],
            para["run_time"],
            para["offset"],
        ]

        save_csv(dmp_file, row_col_dmp, data_list)

        para = para_["kernel"]
        data_list = [para["c"], para["h"]]
        data_list.extend(para["w"].tolist())

        save_csv(kernel_file, row_col_kernel, data_list)

    def find_gait_phase(self, y, dy, ref_path, mm):
        ref_grad = np.gradient(ref_path, self.dt)

        # f, axs = plt.subplots(2, sharex=True)
        # axs[0].plot(ref_path)
        # axs[1].plot(ref_grad)
        # axs[1].axhline(y=0, c='k')
        # plt.show()
        # sys.exit()
        
        # if mm > 70 and mm < 85:
        #     print(mm, y, dy, "   ", ref_path[70:85], ref_grad[70:85])

        min_diff = 1e10
        match_idx = 0
        for i, r in enumerate(ref_path):
            #meaning diff direction, not using < 0 so that we can include places with small gradient
            if dy * ref_grad[i] < 0:
                continue
            
            diff = np.abs(r - y)
            if diff < min_diff:
                min_diff = diff
                match_idx = i

        return match_idx / ref_path.shape[0]

    def plot_step(self, save_pre=None, goal=1.0, tau=1.0):
        """Make step and plot the proceduer."""
        y = np.ones(self.n_dmps) * self.y0
        dy = np.zeros(self.n_dmps)
        num_steps = self.timesteps + 1
        track = np.zeros((self.n_dmps, num_steps))
        T = self.time[-1][-1]


        time = np.arange(0, self.time[0, -1] + self.dt, self.dt)
        eps = 1e-10
        interpolate_time = self.time.ravel()
        interpolate_time[0] -= eps  # Ensure boundary is not exceeded.
        interpolate_time[-1] += eps
        path = np.zeros((self.n_dmps, self.timesteps + 1))
        for d in range(self.n_dmps):
            path_gen = scipy.interpolate.interp1d(interpolate_time, self.path[d, :])
            for t in range(self.timesteps + 1):  # +1 to include the starting point.
                path[d, t] = path_gen(time[t])
        path_temp = path

        # l = np.array([0, 0])
        # h = np.array([-40, 40])
        gp = np.zeros(num_steps)
        err_all = np.zeros(num_steps)

        #Check how the initial data scale the reference
        goal = (self.path_bound[:,1] - y) / (self.path_bound[:,1] - self.path_bound[:,0])
        scaled_offset = (y - self.path_bound[:,0])
        # goal[0] = goal[1]
        # scaled_offset[0] = scaled_offset[1]
        # goal = 1.5
        # scaled_offset = 0
        print("goal_offset", self.goal, "self.offset", self.offset)
        ori_path = path + np.expand_dims(self.offset, 1)
        path = np.multiply(path, np.expand_dims(goal, 1))
        path = path + np.expand_dims(scaled_offset, 1)
        path = path + np.expand_dims(self.offset, 1) #do this after goal scale multiplication because it starts from 0

        increment = scaled_offset / num_steps

        print(path.shape, path_temp.shape)

        y[0] += 0 # initial position
        y[1] += 10
        goal_offset = np.zeros(self.n_dmps)
        new_scale = np.zeros(self.n_dmps)
        goal_offset[0] = 0
        goal_offset[1] = 0
        new_goal = ori_path[:,-1] + goal_offset
        # new_scale[0] = (goal_offset[0] + ori_path[0,-1] - ori_path[0,0])/(ori_path[0,-1] - ori_path[0,0])
        new_scale[0] = 1.0
        new_scale[1] = 1.0
        num_len = path.shape[1]

        ###### -------linear scaling -------- ########
        for j in range(path.shape[0]):
            for i in range(path.shape[1]):
                in_max = ori_path[j, -1]
                in_min = ori_path[j, 0]
                out_max = ori_path[j, -1]
                out_min = y[j]
                # path[j, i] = (ori_path[j, i] - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
                path[j, i] = (ori_path[j, i] - in_min) * new_scale[j] + in_min + (new_goal[j] - in_max) / num_len * i + (y[j] - in_min) / num_len * (num_len - i)

        print("path", path[:,0], path[:, -1], " path_temp ", path_temp[:,0], path_temp[:, -1])
        
        print("path", path[:,0], path[:, -1], " goal ", np.multiply(self.goal, goal))

        ###### -------DMP scaling -------- ########
        for i in range(num_steps):
            gait_phase = self.dt * i / T #+ 0.2
            # gait_phase = (y[1] - l[1]) / (h[1] - l[1])
            # gait_phase = self.find_gait_phase(y[1], dy[1], path[1,:], i)

            # print(i, gait_phase)
            
            if gait_phase > 1:
                gait_phase = 1
            if gait_phase < 0.0:
                gait_phase = 0.0
            gp[i] = gait_phase

            err_c = 0
            err_scale = 0.0
            # if i > 0:
            #     reference = path[:, (int)(num_steps*gait_phase)]
            #     err_c = reference - y
            #     err_c = err_c[1] * err_scale
            # else:
            #     err_c = 0
            
            # self.ay = ay_ori * (1 + err_c)
            # self.by = by_ori * (1 + err_c)
            y_prev = deepcopy(y)
            y, dy, ddy = self.step_real(gait_phase, y, dy, scale=new_scale, goal_offset=goal_offset, tau=tau, extra_force=0.0)
            # y = y - increment
            
            # if i > 1:
            #     dy = (y - y_prev) / self.dt
            
            track[:, i] = deepcopy(y)

        beforeTargetCnt = (int)(gp[0] * num_steps)
        appdArr = np.ones((self.n_dmps, beforeTargetCnt))
        for i in range(self.n_dmps):
            appdArr[i, :] = 0 #track[i, 0]
        track = np.concatenate([appdArr, track], axis=1)
        gp = np.concatenate([appdArr[0], gp])
        # print(track.shape)
        # print(path.shape)
        # err_all = track[1,:500] - path[1, :500]

        # print(err_all[:3])

        f, axs = plt.subplots(5, sharex=True)
        f.set_size_inches(10,10)
        axs[0].plot(gp[:], label="gait phase")
        axs[1].set_title("Right")
        axs[1].plot(path[0, :], label="Linear scaling")
        axs[1].plot(track[0, :], label="DMP Output")
        axs[1].plot(ori_path[0, :], "--", label="Original Reference")
        # axs[1].axhline(y=y_ini[0], c="r")
        axs[2].set_title("Left")
        axs[2].plot(path[1, :], label="Linear scaling")
        axs[2].plot(track[1, :], label="DMP Output")
        axs[2].plot(ori_path[1, :], "--", label="Original Reference")
        # axs[2].axhline(y=y_ini[1], c="r")
        # axs[3].plot(err_all, label="error")
        axs[3].plot(path[0, :] - ori_path[0, :], label="linear err ")
        axs[3].plot(track[0, :] - ori_path[0, :], label="dmp err ")
        axs[4].plot(path[1, :] - ori_path[1, :], label="linear err ")
        axs[4].plot(track[1, :] - ori_path[1, :], label="dmp err ")
        for ax in axs:
            ax.legend()
            ax.axhline(y=0, c="k")
        # track_err = np.sqrt(np.mean(err_all[beforeTargetCnt+1:500]**2))
        # axs[0].set_title("Tracking Err: {:.2f}".format(track_err))

        # plt.savefig("error_coupling{}.png".format(err_scale))

        # for i in range(self.n_dmps):
        #     plt.figure()
        #     plt.plot(path[i, 1:] + self.offset[i], "--")
        #     plt.plot(track[i, :])
        #     plt.legend(["given", "step"])
        #     plt.xlabel("number of samples")
        #     plt.ylabel("displacement")

        #     if save_pre is not None:
        #         plt.savefig(save_pre + f"step_{i}.png")
