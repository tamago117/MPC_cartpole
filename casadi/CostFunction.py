# reference https://gist.github.com/mayataka/7e608dbbbcd93d232cf44e6cab6b0332
import math

class CostFunction:
    def __init__(self):
        self.nx = 4
        self.nu = 1
        # stage cost
        self.Q  = [2.0, 5.0, 0.05, 0.05]       # state weights
        self.R  = [0.01]                         # input weights
        # terminal cost
        self.Qf = [2.0, 5.0, 0.05, 0.05]       # terminal state weights

    def stage_cost(self, x, u, x_ref):
        cost = 0
        for i in range(self.nx):
            cost += 0.5 * self.Q[i] * (x[i] - x_ref[i])**2
        for i in range(self.nu):
            cost += 0.5 * self.R[i] * u[i]**2
        return cost

    def terminal_cost(self, x, x_ref):
        cost = 0
        for i in range(self.nx):
            cost += 0.5 * self.Q[i] * (x[i] - x_ref[i])**2
        return cost