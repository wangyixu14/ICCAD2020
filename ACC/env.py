import matplotlib.pyplot as plt
import numpy as np
from numpy import linalg as LA
import math
import random
import math
import gym


def main():
    env = ACC()
    for ep in range(1):
        state = env.reset()
        for i in range(env.max_iteration):
            action = np.random.uniform(low=-1, high=1)
            next_state, reward, done, disturbance = env.step(action)
            print(state, action, next_state, reward, done, disturbance)
            state = next_state
            if done:
                break

class ACC:
    k = 0.2
    deltaT = 0.1
    gap_eqb = 150
    safe_range_low = -30
    safe_range_high = 30
    v_eqb = 40
    safe_v_range = 15

    u_range = 20

    period_w = 5

    nx = 2
    nu = 2

    max_iteration = 100
    error = 1e-5

    frontVehicleVMean = 40
    frontVehicleVRange = 0

    def __init__(self, gap=None, v=None):
        if gap is None or v is None:
            gap = np.random.uniform(low=self.safe_range_low, high=self.safe_range_high) + self.gap_eqb
            v = np.random.rand(1)[0] * self.safe_v_range * 2 - self.safe_v_range + self.v_eqb
            self.gap = gap
            self.v = v
        else:
            self.gap = gap
            self.v = v
        
        self.t = 0
        self.state = np.array([[self.gap], [self.v]])

        self.vf_t = self.v_eqb 
        self.state_trace = [self.state]
        self.w_trace = [0]
        self.control_trace = []
        self.u_last = 0

    def reset(self, gap=None, v=None):
        if gap is None or v is None:
            # random_init = np.random.rand(2)
            gap = np.random.uniform(low=self.safe_range_low, high=self.safe_range_high) + self.gap_eqb
            v = np.random.rand(1)[0] * self.safe_v_range * 2 - self.safe_v_range + self.v_eqb
            self.gap = gap
            self.v = v
        else:
            self.gap = gap
            self.v = v
        
        self.t = 0
        self.state = np.array([[self.gap], [self.v]])

        self.vf_t = self.v_eqb 
        self.state_trace = [self.state]
        self.w_trace = [0]
        self.control_trace = []
        self.u_last = 0
        return self.normalize_state(self.state)

    def step(self, action):
        disturbance = np.random.uniform(-4, 4)
        # disturbance = 0
        u = action * self.u_range
        self.w_w = self.frontVehicleVRange * np.sin(2 * np.pi / self.period_w * (self.t) * self.deltaT) + self.frontVehicleVMean - self.vf_t
        self.w_trace.append(self.w_w+self.vf_t - self.frontVehicleVMean)
        A, B, C = self.get_model_matrix()
    
        # assert False
        uu = np.array([[self.vf_t + disturbance],[u]])
        ww = np.array([[0],[self.w_w]])
        # ww = np.array([[0],[disturbance]])
        self.state = np.dot(A, self.state) + np.dot(B, uu) + np.dot(C, ww) 
        self.state_trace.append(self.state)
        self.control_trace.append(u)

        self.vf_t += self.w_w
        self.t = self.t + 1
        done = self.if_unsafe() or self.t == self.max_iteration
        reward = self.design_reward(u, self.u_last, smoothness=0.5)
        self.u_last = u

        return self.normalize_state(self.state), reward, done, disturbance

    def design_reward(self, u, u_last, smoothness):
        r = 0
        # r -= 0.5*abs(self.state[0, 0] - self.gap_eqb)
        # r -= abs(self.state[1, 0] - self.v_eqb)
        # # r -= 0.5 * abs(u)
        # r -= smoothness * abs(u - u_last)
        # if self.if_unsafe():
        #     r -= 40
        # else:
        #     r += 25      
        return r

    def if_unsafe(self):
        if self.state[0, 0] < self.gap_eqb + self.safe_range_low - self.error or self.state[0, 0] > self.gap_eqb + self.safe_range_high + self.error or self.state[1, 0] < self.v_eqb - self.safe_v_range - self.error or self.state[1, 0] > self.v_eqb + self.safe_v_range + self.error:
            return 1
        else:
            return 0

    def normalize_state(self, state):
        normal_gap = state[0,0] - self.gap_eqb
        normal_v = state[1,0] - self.v_eqb
        return np.reshape(np.array([normal_gap, normal_v]), (2,))

    def get_model_matrix(self):
        # Model Parameter
        A = np.array([
            [0.0, -1.0],
            [0.0, -self.k]
        ])

        A = np.eye(self.nx) + self.deltaT * A

        B = np.eye(self.nu)
        B = self.deltaT * B

        C = np.array([
            [0.0, 1.0],
            [0.0, 0.0]
        ])
        C = self.deltaT * C

        D = np.array([
            0.0,
            0.0
        ])

        return A, B, C


if __name__ == '__main__':
    main()
