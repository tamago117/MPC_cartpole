#pragma once

#include <math.h>
#include <vector>
#include <eigen3/Eigen/Dense>

namespace CartPole
{
    __device__ __host__ Eigen::VectorXf dynamics(Eigen::VectorXf x_vec, Eigen::VectorXf u_vec, float dt);
}