import math
from threading import current_thread
import time
import matplotlib.pyplot as plt
import numpy as np
from casadi import *

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ocp_get_default_cmake_builder
from CartPole import CartPole
from CostFunction import CostFunction

class MPC:
    def __init__(self, config):
        self.T = config['DT']*config['HORIZON'] # horizon length
        self.N = config['HORIZON'] # discreate grid number
        self.dt = config['DT'] # minute time
        self.Q  = [config['Q0'], config['Q1'], config['Q2'], config['Q3']]       # state weights
        self.Qf = [config['Qf0'], config['Qf1'], config['Qf2'], config['Qf3']]       # terminal state weights
        self.R  = [config['R0']]                       # input weights

        # model definition
        cartpole = CartPole()

        # optimal control problem definition
        mpc = AcadosOcp()
        mpc.solver_options.tf = self.T # horizon length
        mpc.dims.N = self.N
        mpc.model = cartpole.model()

        # cost function definition
        mpc.cost.cost_type = 'LINEAR_LS' # quadratic form stage cost
        mpc.cost.cost_type_e = 'LINEAR_LS' # quadratic form terminal cost
        nx = mpc.model.x.size()[0]
        nu = mpc.model.u.size()[0]
        ny = nx + nu

        mpc.cost.W_e = np.diag(self.Qf) # terminal cost matrix
        mpc.cost.W = np.diag(np.concatenate([self.Q, self.R])) # stage cost matrix
        mpc.cost.Vx = np.zeros((ny, nx))
        mpc.cost.Vx[:nx,:nx] = np.eye(nx)
        mpc.cost.Vx_e = np.eye(nx)
        Vu = np.zeros((ny, nu))
        Vu[4,0] = 1.0
        mpc.cost.Vu = Vu

        #ref
        mpc.cost.yref  = np.concatenate([np.array([0, 3.14, 0, 0]), np.zeros(1)])
        mpc.cost.yref_e = np.concatenate([np.array([0, 3.14, 0, 0])])

        # constraints definition
        mpc.constraints.constr_type = 'BGH' # box constraints
        mpc.constraints.x0 = np.array([0, 0, 0, 0]) # initial state
        mpc.constraints.lbu = np.array([-config['MAX_INPUT']]) # lower bound input
        mpc.constraints.ubu = np.array([config['MAX_INPUT']]) # upper bound input
        mpc.constraints.idxbu = np.array([0]) # index of input
        mpc.constraints.lbx = np.array([-config['MAX_X']]) # lower bound state
        mpc.constraints.ubx = np.array([config['MAX_X']]) # upper bound state
        mpc.constraints.idxbx = np.array([0]) # index of state


        # solver options
        mpc.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # Riccati recursion
        mpc.solver_options.hessian_approx = 'GAUSS_NEWTON'
        mpc.solver_options.integrator_type = 'ERK'
        mpc.solver_options.sim_method_num_stages = 1 # ERK + 1 でオイラー法， ERK + 4 でよくある runge-kutta法
        # mpc.solver_options.globalization = 'FIXED_STEP' # 直線探索をオフに
        mpc.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP # 反復回数1回（nlp_solver_max_iter = 1）であれば SQP_RTIが有用．詳しくは"Real-time iteration scheme"で検索．
        # mpc.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP # 反復回数1回（nlp_solver_max_iter = 1）であれば SQP_RTIが有用．詳しくは"Real-time iteration scheme"で検索．
        mpc.solver_options.nlp_solver_max_iter = config['ITERATIONS'] # 反復回数をリアルタイム計算可能な程度に固定
        mpc.solver_options.qp_solver_iter_max = config['ITERATIONS'] # 反復回数をリアルタイム計算可能な程度に固定
        mpc.solver_options.hpipm_mode = 'SPEED'

        # create solver
        cmake_builder = ocp_get_default_cmake_builder()
        # mpc_solver = AcadosOcpSolver(mpc, json_file = 'acados_mpc.json') # Pythonで解く場合
        self.mpc_solver = AcadosOcpSolver(mpc, json_file = 'acados_mpc.json', cmake_builder=cmake_builder) # CMakeでコード生成＋解く場合
        # mpc_solver.options_set('qp_tau_min', 1.0e-01)
        # mpc_solver.options_set('warm_start_first_qp', 1)
        # mpc_solver.options_set('qp_tol_stat', 1.0e-04)
        # mpc_solver.options_set('qp_tol_eq', 1.0e-04)
        # mpc_solver.options_set('qp_tol_ineq', 1.0e-04)
        # mpc_solver.options_set('qp_tol_comp', 1.0e-04)

    def init(self, x0, ref, warm_start_num=1):
        # x0: initial state
        # ref: control reference
        # warm_start_num: number of warm start

        # update reference
        for j in range(self.N):
            yref  = np.concatenate([ref, np.zeros(1)])
            self.mpc_solver.set(j, "yref", yref)
        yref_N  = ref
        self.mpc_solver.set(self.N, "yref", yref_N)

        for i in range(warm_start_num):
            self.mpc_solver.constraints_set(0, "lbx",  x0)
            self.mpc_solver.constraints_set(0, "ubx",  x0)
            self.mpc_solver.solve()

    def solve(self, x, ref):
        # x: state
        # ref: reference trajectory
        # return: control input
        # update yref

        # update reference
        for j in range(self.N):
            yref  = np.concatenate([ref, np.zeros(1)])
            self.mpc_solver.set(j, "yref", yref)
        yref_N  = ref
        self.mpc_solver.set(self.N, "yref", yref_N)

        self.mpc_solver.constraints_set(0, "lbx",  x)
        self.mpc_solver.constraints_set(0, "ubx",  x)
        status = self.mpc_solver.solve()
        u = self.mpc_solver.get(0, "u")

        return u

