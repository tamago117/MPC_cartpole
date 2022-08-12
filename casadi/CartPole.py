from casadi import *
import math

class CartPole:
    def __init__(self):
        self.mc = 2.0 #cart mass
        self.mp = 0.2 #pole mass
        self.l = 0.5 #pole length
        self.ga = 9.81 #gravity constant

    def dynamics(self, x, u):
        y = x[0] #cart position[m]
        th = x[1] #pole angle[rad]
        dy = x[2] #cart velocity[m/s]
        dth = x[3] #pole angle velocity[rad/s]
        f = u[0] #input[N]

        #cart acceleration
        ddy = (f+mp*sin(th)*(l*dth*dth+ga*cos(th))) / (mc+mp*sin(th)*sin(th))
        # pole angle acceleration
        ddth = (-f*cos(th)-mp*l*dth*dth*cos(th)*sin(th)-(mc+mp)*ga*sin(th)) / (l * (mc+mp*sin(th)*sin(th)))
        
        return dy, dth, ddy, ddth