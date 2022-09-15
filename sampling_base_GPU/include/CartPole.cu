#include "CartPole.cuh"

__device__ __host__ Eigen::VectorXf dynamics(Eigen::VectorXf x_vec, Eigen::VectorXf u_vec, float dt)
{
    const float cart_mass = 2.0;
    const float pole_mass = 0.2;
    const float pole_length = 0.5;
    const float gravity = 9.81;

    Eigen::VectorXf next_x(4);

    float x = x_vec(0); // cart position[m]
    float theta = x_vec(1); // pole angle[rad]
    float dx = x_vec(2); // cart velocity[m/s]
    float dtheta = x_vec(3); // pole angle velocity[rad/s]
    float f = u_vec(0); //input[N]

    // cart acceleration
    float ddx = (f+pole_mass*sin(theta)*(pole_length*dtheta*dtheta + gravity*cos(theta)))/(cart_mass+pole_mass*sin(theta)*sin(theta));

    float ddtheta = (-f*cos(theta)-pole_mass*pole_length*dtheta*dtheta*cos(theta)*sin(theta) - (cart_mass+pole_mass)*gravity*sin(theta))/(pole_length*(cart_mass+pole_mass*sin(theta)*sin(theta)));

    next_x(0) = x + dx*dt;
    next_x(1) = theta + dtheta*dt;
    next_x(2) = dx + ddx*dt;
    next_x(3) = dtheta + ddtheta*dt;

    return next_x;
}