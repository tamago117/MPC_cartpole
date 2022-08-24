# reference https://gist.github.com/mayataka/7e608dbbbcd93d232cf44e6cab6b0332

from casadi import *
import math
import numpy as np

from CostFunction import CostFunction
from CartPole import CartPole

class MPC:
    def __init__(self):
        T = 1.0 # horizon length
        N = 20 # discreate grid number
        dt = T/N # minute time
        nx = 4 # state variable number
        nu = 1 # input variable number
        cartpole = CartPole()
        cost_function = CostFunction()

        w = [] # contain optimal variable
        w0 = [] # contain initial optimal variable
        lbw = [] # lower bound optimal variable
        ubw = [] # upper bound optimal variable
        J = 0 # cost function
        g  = [] # constrain
        lbg = [] # lower bound constrain
        ubg = [] # upper bound constrain
        lam_x0 = [] # Lagrangian multiplier
        lam_g0 = [] # Lagrangian multiplier

        Xk = MX.sym('X0', nx) # initial time state vector x0
        w += [Xk]
        # equality constraint
        lbw += [0, 0, 0, 0]  # constraints are set by setting lower-bound and upper-bound to the same value
        ubw += [0, 0, 0, 0]      # constraints are set by setting lower-bound and upper-bound to the same value
        w0 +=  [0, 0, 0, 0]      # x0 initial estimate
        lam_x0 += [0, 0, 0, 0]    # Lagrangian multiplier initial estimate

        for k in range(N):
            Uk = MX.sym('U_' + str(k), nu)
            w += [Uk]
            lbw += [-25.0]
            ubw += [25.0]
            w0 += [0]
            lam_x0 += [0]

            #stage cost
            J = J + cost_function.stage_cost(Xk, Uk)

            # Discretized equation of state by forward Euler
            dXk = cartpole.dynamics(Xk, Uk)
            Xk_next = vertcat(Xk[0] + dXk[0] * dt,
                              Xk[1] + dXk[1] * dt,
                              Xk[2] + dXk[2] * dt,
                              Xk[3] + dXk[3] * dt)
            Xk1 = MX.sym('X_' + str(k+1), nx)
            w   += [Xk1]
            lbw += [-1.0, math.pi-1.0, -inf, -inf]
            ubw += [1.0, math.pi+1.0, inf, inf]
            w0 += [0.0, 0.0, 0.0, 0.0]
            lam_x0 += [0, 0, 0, 0]

            # (xk+1=xk+fk*dt) is introduced as an equality constraint
            g   += [Xk_next-Xk1]
            lbg += [0, 0, 0, 0]     # Equality constraints are set by setting lower-bound and upper-bound to the same value
            ubg += [0, 0, 0, 0]     # Equality constraints are set by setting lower-bound and upper-bound to the same value
            lam_g0 += [0, 0, 0, 0]
            Xk = Xk1

        # finite cost
        J = J + cost_function.terminal_cost(Xk)

        self.J = J
        self.w = vertcat(*w)
        self.g = vertcat(*g)
        self.x = w0
        self.lam_x = lam_x0
        self.lam_g = lam_g0
        self.lbx = lbw
        self.ubx = ubw
        self.lbg = lbg
        self.ubg = ubg

        # 非線形計画問題(NLP)
        self.nlp = {'f': self.J, 'x': self.w, 'g': self.g} 
        # Ipopt ソルバー，最小バリアパラメータを0.1，最大反復回数を5, ウォームスタートをONに
        self.solver = nlpsol('solver', 'ipopt', self.nlp, {'calc_lam_p':True, 'calc_lam_x':True, 'print_time':False, 'ipopt':{'max_iter':5, 'mu_min':0.1, 'warm_start_init_point':'yes', 'print_level':0, 'print_timing_statistics':'no'}})
        # self.solver = nlpsol('solver', 'scpgen', self.nlp, {'calc_lam_p':True, 'calc_lam_x':True, 'qpsol':'qpoases', 'print_time':False, 'print_header':False, 'max_iter':5, 'hessian_approximation':'gauss-newton', 'qpsol_options':{'print_out':False, 'printLevel':'none'}}) # print をオフにしたいがやり方がわからない

    def init(self, x0=None):
        if x0 is not None:
            # 初期状態についての制約を設定
            self.lbx[0:4] = x0
            self.ubx[0:4] = x0
        # primal variables (x) と dual variables（ラグランジュ乗数）の初期推定解も与えつつ solve（warm start）
        sol = self.solver(x0=self.x, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # 次の warm start のために解を保存
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()

    def solve(self, x0):
        # 初期状態についての制約を設定
        nx = x0.shape[0]
        self.lbx[0:nx] = x0
        self.ubx[0:nx] = x0
        # primal variables (x) と dual variables（ラグランジュ乗数）の初期推定解も与えつつ solve（warm start）
        sol = self.solver(x0=self.x, lbx=self.lbx, ubx=self.ubx, lbg=self.lbg, ubg=self.ubg, lam_x0=self.lam_x, lam_g0=self.lam_g)
        # 次の warm start のために解を保存
        self.x = sol['x'].full().flatten()
        self.lam_x = sol['lam_x'].full().flatten()
        self.lam_g = sol['lam_g'].full().flatten()
        return np.array([self.x[4]]) # 制御入力を return