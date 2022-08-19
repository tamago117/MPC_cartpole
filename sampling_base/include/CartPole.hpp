#pragma once

#include <math.h>
#include <vector>
#include <eigen3/Eigen/Dense>

class CartPole
{
    public:
        CartPole(double cart_mass_, double pole_mass_, double pole_length_, double dt_);
        Eigen::VectorXd dynamics(Eigen::VectorXd x_vec, Eigen::VectorXd u_vec);

    private:
        double cart_mass;
        double pole_mass;
        double pole_length;
        double dt;
        const double gravity = 9.81;
};

CartPole::CartPole(double cart_mass_, double pole_mass_, double pole_length_, double dt_)
{
    cart_mass = cart_mass_;
    pole_mass = pole_mass_;
    pole_length = pole_length_;
    dt = dt_;
}

Eigen::VectorXd CartPole::dynamics(Eigen::VectorXd x_vec, Eigen::VectorXd u_vec)
{
    Eigen::VectorXd next_x(4);

    double x = x_vec(0); // cart position[m]
    double theta = x_vec(1); // pole angle[rad]
    double dx = x_vec(2); // cart velocity[m/s]
    double dtheta = x_vec(3); // pole angle velocity[rad/s]
    double f = u_vec(0); //input[N]

    // cart acceleration
    double ddx = (f+pole_mass*sin(theta)*(pole_length*dtheta*dtheta + gravity*cos(theta)))/(cart_mass+pole_mass*sin(theta)*sin(theta));

    double ddtheta = (-f*cos(theta)-pole_mass*pole_length*dtheta*dtheta*cos(theta)*sin(theta) - (cart_mass+pole_mass)*gravity*sin(theta))/(pole_length*(cart_mass+pole_mass*sin(theta)*sin(theta)));

    next_x(0) = x + dx*dt;
    next_x(1) = theta + dtheta*dt;
    next_x(2) = dx + ddx*dt;
    next_x(3) = dtheta + ddtheta*dt;

    return next_x;
}