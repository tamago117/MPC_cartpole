class CostFunction:
    def __init__(self):
        self.nx = 4
        self.nu = 1
        self.x_ref = [0.0, pi, 0.0, 0.0]   # target
        # stage cost
        self.Q  = [2.5, 10.0, 0.01, 0.01]       # state weights
        self.R  = [0.1]                         # input weights
        # terminal cost
        self.Qf = [2.5, 10.0, 0.01, 0.01]       # terminal state weights

    def stage_cost(self, x, u):
        cost = 0
        for i in range(self.nx):
            cost += 0.5 * self.Q[i] * (x[i]-self.x_ref[i])**2
        for i in range(self.nu):
            cost += 0.5 * self.R[i] * u[i]**2
        return cost

    def terminal_cost(self, x, u):
        cost = 0
        for i in range(self.nx):
            cost += 0.5 * self.Q[i] * (x[i]-self.x_ref[i])**2
        return cost