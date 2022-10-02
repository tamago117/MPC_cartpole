#pragma once

#include <math.h>
#include "MCMPC_config.cuh"

namespace CartPole
{
    __host__ __device__ vectorF<NX> dynamics(vectorF<NX> x_vec, vectorF<NU> u_vec, float dt);

}