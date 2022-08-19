import math
from threading import current_thread
import time
import matplotlib.pyplot as plt
import numpy as np
from casadi import *

from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel, ocp_get_default_cmake_builder
from CartPole import CartPole
from CostFunction import CostFunction

def main():
    model = AcadosModel()
    y = SX.sym('y')
    th = SX.sym('th')
    dy = SX.sym('dy')
    dth = SX.sym('dth')
    x = vertcat(y, th, dy, dth)
    u = vertcat(SX.sym('f'))

    y_dot = SX.sym('y_dot')
    th_dot = SX.sym('th_dot')
    dy_dot = SX.sym('dy_dot')
    dth_dot = SX.sym('dth_dot')
    xdot = vertcat(y_dot, th_dot, dy_dot, dth_dot)

    cartpole = CartPole()
    dy, dth, ddy, ddth = cartpole.dynamics(x, u)
    f = vertcat(dy, dth, ddy, ddth)

    model.x = x
    model.u = u
    model.p = [] #時変パラメータ？
    model.z = []
    model.f_expl_expr = f # 状態方程式
    model.f_impl_expr = xdot - f # DAEモデルの記述用？普通の x\dot = f(x, u) は左のように記述
    model.xdot = xdot # xdot の CasADi symbols
    model.name = f'cart_pole'




    # MPC用最適制御問題
    mpc = AcadosOcp()
    mpc.solver_options.tf = 1.0 # horizon length
    mpc.dims.N = 20
    mpc.model = model

    ## 評価関数
    mpc.cost.cost_type = 'LINEAR_LS' # ステージコストを2次形式に指定
    mpc.cost.cost_type_e = 'LINEAR_LS' # 終端コストを2次形式に指定
    nx = mpc.model.x.size()[0]
    nu = mpc.model.u.size()[0]
    ny = nx + nu

    cost_function = CostFunction() # 以前作成したクラス CostFunctionを再利用

    mpc.cost.W_e = np.diag(cost_function.Qf) # 終端コストのヘッセ行列
    mpc.cost.W = np.diag(np.concatenate([cost_function.Q, cost_function.R])) # ステージコスト

    mpc.cost.Vx = np.zeros((ny, nx))
    mpc.cost.Vx[:nx,:nx] = np.eye(nx)

    Vu = np.zeros((ny, nu))
    Vu[4,0] = 1.0
    mpc.cost.Vu = Vu

    mpc.cost.Vx_e = np.eye(nx)

    mpc.cost.yref  = np.concatenate([cost_function.x_ref, np.zeros(1)])
    mpc.cost.yref_e = np.concatenate([cost_function.x_ref])


    ## 制約
    mpc.constraints.x0 = np.array([0.0, 0.0, 0.0, 0.0]) #初期時刻の状態を制約として指定

    mpc.constraints.constr_type = 'BGH'
    mpc.constraints.lbu = np.array([-25.0]) #制御入力の下限
    mpc.constraints.ubu = np.array([25.0]) #制御入力の上限
    mpc.constraints.idxbu = np.array([0]) #制御入力の制約をかけるインデックス

    mpc.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM' # Riccati recursion
    mpc.solver_options.hessian_approx = 'GAUSS_NEWTON'
    mpc.solver_options.integrator_type = 'ERK'
    mpc.solver_options.sim_method_num_stages = 1 # ERK + 1 でオイラー法， ERK + 4 でよくある runge-kutta法
    # mpc.solver_options.globalization = 'FIXED_STEP' # 直線探索をオフに
    mpc.solver_options.nlp_solver_type = 'SQP_RTI' # SQP_RTI, SQP # 反復回数1回（nlp_solver_max_iter = 1）であれば SQP_RTIが有用．詳しくは"Real-time iteration scheme"で検索．
    # mpc.solver_options.nlp_solver_type = 'SQP' # SQP_RTI, SQP # 反復回数1回（nlp_solver_max_iter = 1）であれば SQP_RTIが有用．詳しくは"Real-time iteration scheme"で検索．
    mpc.solver_options.nlp_solver_max_iter = 5 # 反復回数をリアルタイム計算可能な程度に固定
    mpc.solver_options.qp_solver_iter_max = 5 # 反復回数をリアルタイム計算可能な程度に固定
    mpc.solver_options.hpipm_mode = 'SPEED'

    cmake_builder = ocp_get_default_cmake_builder()
    # mpc_solver = AcadosOcpSolver(mpc, json_file = 'acados_mpc.json') # Pythonで解く場合
    mpc_solver = AcadosOcpSolver(mpc, json_file = 'acados_mpc.json', cmake_builder=cmake_builder) # CMakeでコード生成＋解く場合
    # mpc_solver.options_set('qp_tau_min', 1.0e-01)
    # mpc_solver.options_set('warm_start_first_qp', 1)
    # mpc_solver.options_set('qp_tol_stat', 1.0e-04)
    # mpc_solver.options_set('qp_tol_eq', 1.0e-04)
    # mpc_solver.options_set('qp_tol_ineq', 1.0e-04)
    # mpc_solver.options_set('qp_tol_comp', 1.0e-04)

    sim_time = 10.0 # 10秒間のシミュレーション
    sampling_time = 0.01 # 0.01秒（10ms）のサンプリング周期
    sim_steps = math.floor(sim_time/sampling_time)
    xs = []
    us = []
    cartpole = CartPole()

    x = np.zeros(4)
    mpc_solver.constraints_set(0, "lbx",  x)
    mpc_solver.constraints_set(0, "ubx",  x)

    WARM_START_ITERS = 100
    for i in range(WARM_START_ITERS):
        mpc_solver.solve()

    for step in range(sim_steps):
        if step%(1/sampling_time)==0:
            print('t =', step*sampling_time)
        # 初期状態を更新
        mpc_solver.constraints_set(0, "lbx",  x)
        mpc_solver.constraints_set(0, "ubx",  x)
        current_time = time.time()
        status = mpc_solver.solve()
        print(str(math.floor(1/(time.time() - current_time))) + "hz")
        # 制御入力をソルバーから取得
        u = mpc_solver.get(0, "u")
        xs.append(x)
        us.append(u)
        x1 = x + sampling_time * np.array(cartpole.dynamics(x, u))
        x = x1

        plot_cart(x[0], x[1])

    # シミュレーション結果をプロット
    xs1 = [x[0] for x in xs]
    xs2 = [x[1] for x in xs]
    xs3 = [x[2] for x in xs]
    xs4 = [x[3] for x in xs]
    tgrid = [sampling_time*k for k in range(sim_steps)]

    plt.figure(1)
    plt.clf()
    plt.plot(tgrid, xs1, '--')
    plt.plot(tgrid, xs2, '-')
    plt.plot(tgrid, xs3, '-')
    plt.plot(tgrid, xs4, '-')
    plt.step(tgrid, us, '-.')
    plt.xlabel('t')
    plt.legend(['y(x1)','th(x2)', 'dy(x3)', 'dth(x4)','u'])
    plt.grid()
    plt.show()

def plot_cart(xt, theta):
    theta += math.pi
    l_bar = 1.0  # length of bar
    cart_w = 0.5
    cart_h = 0.25
    radius = 0.05

    cx = np.array([-cart_w / 2.0, cart_w / 2.0, cart_w /
                   2.0, -cart_w / 2.0, -cart_w / 2.0])
    cy = np.array([0.0, 0.0, cart_h, cart_h, 0.0])
    cy += radius * 2.0

    cx = cx + xt

    bx = np.array([0.0, l_bar * math.sin(-theta)])
    bx += xt
    by = np.array([cart_h, l_bar * math.cos(-theta) + cart_h])
    by += radius * 2.0

    angles = np.arange(0.0, math.pi * 2.0, math.radians(3.0))
    ox = np.array([radius * math.cos(a) for a in angles])
    oy = np.array([radius * math.sin(a) for a in angles])

    rwx = np.copy(ox) + cart_w / 4.0 + xt
    rwy = np.copy(oy) + radius
    lwx = np.copy(ox) - cart_w / 4.0 + xt
    lwy = np.copy(oy) + radius

    wx = np.copy(ox) + bx[-1]
    wy = np.copy(oy) + by[-1]

    plt.clf()

    plt.plot(np.array(cx).flatten(), np.array(cy).flatten(), "-b")
    plt.plot(np.array(bx).flatten(), np.array(by).flatten(), "-k")
    plt.plot(np.array(rwx).flatten(), np.array(rwy).flatten(), "-k")
    plt.plot(np.array(lwx).flatten(), np.array(lwy).flatten(), "-k")
    plt.plot(np.array(wx).flatten(), np.array(wy).flatten(), "-k")
    plt.title(f"x: {xt:.2f} , theta: {math.degrees(theta):.2f}")

    # for stopping simulation with the esc key.
    plt.gcf().canvas.mpl_connect(
        'key_release_event',
        lambda event: [exit(0) if event.key == 'escape' else None])

    plt.axis("equal")

    #plt.xlim([-5.0, 2.0])
    plt.pause(0.001)

if __name__ == '__main__':
    main()