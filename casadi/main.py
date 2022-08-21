# reference https://gist.github.com/mayataka/7e608dbbbcd93d232cf44e6cab6b0332

import math
import time
import matplotlib.pyplot as plt
import numpy as np
from casadi import *

from MPC import MPC
from CostFunction import CostFunction
from CartPole import CartPole

sim_time = 10.0
sampling_time = 0.01 # 100hz
sim_steps = math.floor(sim_time / sampling_time)

l_bar = 0.5  # length of bar

def main():
    xs = []
    us = []
    cartpole = CartPole()
    mpc = MPC()
    #mpc.init()
    x = np.array([0.0, 0.0, 0.0, 0.0])
    x_ref = np.array([0.0, math.pi, 0.0, 0.0])   # target

    for step in range(sim_steps):
        if step%(1/sampling_time) == 0:
            print('t=', step*sampling_time)

        if step*sampling_time>5.0:
            x_ref = np.array([0.5, math.pi, 0.0, 0.0])

        if step*sampling_time>7.0:
            x_ref = np.array([-0.5, math.pi, 0.0, 0.0])

        current_time = time.time()
        u = mpc.solve(x, x_ref)
        print(str(math.floor(1/(time.time() - current_time))) + "hz")
        xs.append(x)
        us.append(u)
        x1 = x + sampling_time * np.array(cartpole.dynamics(x, u))
        x = x1

        plot_cart(x[0], x[1])

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
    cart_w = 0.25
    cart_h = 0.12
    radius = 0.03

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