#include "CartPole.cuh"

__host__ __device__ vectorF<NX> CartPole::dynamics(vectorF<NX> x_vec, vectorF<NU> u_vec, float dt)
{
    const float cart_mass = 2.0;
    const float pole_mass = 0.2;
    const float pole_length = 0.5;
    const float gravity = 9.81;

    vectorF<4> next_x;

    float x = x_vec.vector[0]; // cart position[m]
    float theta = x_vec.vector[1]; // pole angle[rad]
    float dx = x_vec.vector[2]; // cart velocity[m/s]
    float dtheta = x_vec.vector[3]; // pole angle velocity[rad/s]
    float f = u_vec.vector[0]; //input[N]

    // cart acceleration
    float ddx = (f+pole_mass*sin(theta)*(pole_length*dtheta*dtheta + gravity*cos(theta)))/(cart_mass+pole_mass*sin(theta)*sin(theta));

    float ddtheta = (-f*cos(theta)-pole_mass*pole_length*dtheta*dtheta*cos(theta)*sin(theta) - (cart_mass+pole_mass)*gravity*sin(theta))/(pole_length*(cart_mass+pole_mass*sin(theta)*sin(theta)));

    next_x.vector[0] = x + dx*dt;
    next_x.vector[1] = theta + dtheta*dt;
    next_x.vector[2] = dx + ddx*dt;
    next_x.vector[3] = dtheta + ddtheta*dt;

    return next_x;
}