#pragma once

#include <map>
#include <vector>
#include <string>
#include <random>
#include <algorithm>

// cuda
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>
#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>
#include <thrust/reduce.h>
#include "cuda_utils.cuh"

#include "MCMPC_config.cuh"
#include "CartPole.cuh"

class MCMPC_CartPole
{
public:
    MCMPC_CartPole();
    ~MCMPC_CartPole();
    vectorF<NU> solve(const vectorF<NX> &target_state, const vectorF<NX> &current_state);

private:
    // mpc
    //static const int NX = 4; // state = x angle v angleVel
    //static const int NU = 1; // input = v

    vectorF<NU> R;  // input weights
    vectorF<NU> Rd; // input difference weights
    vectorF<NX> Q;  // state weights
    vectorF<NX> Qf; // final state weights
    // tranfer value
    float input_weight, input_diff_weight, pos_weight, angle_weight, v_weight, angleVelocity_weight;

    // parallel computing variable
    
    thrust::device_vector<u_array> u_array_device;
    thrust::host_vector<u_array> u_array_host;
    thrust::device_vector<vectorF<NU>> input_list_device;
    thrust::host_vector<vectorF<NU>> input_list_host;
    thrust::device_vector<x_array> x_array_device;
    
    thrust::device_vector<float> cost_array_device;
    thrust::host_vector<float> cost_array_host;
    thrust::device_vector<int> indices_device;
    thrust::host_vector<int> indices_host;

    curandState *random_seed;

    // GPU config
    unsigned int NUM_BLOCKS;

    const float COST_OVER_VALUE = 10000;

    // constrait

    //mcmpc config
    MCMPC_config config;

};

// cuda functions
//__device__ __host__ vectorF<NX> dynamics(vectorF<NX> x_vec, vectorF<NU> u_vec, float dt);
__global__ void ParallelMonteCarloSimulation(u_array *u_array, float *cost_array, const vectorF<NX> xref, const vectorF<NX> current_state, x_array *x_array, vectorF<NU> *mean, curandState *random_seed);

__device__ float input_constrain(const float _input, const float lower_bound, const float upper_bound);
__device__ float state_constrain(const float _state, const float lower_bound, const float upper_bound);
__device__ float GenerateRadomInput(curandState *random_seed, unsigned int id, float mean, float variance);