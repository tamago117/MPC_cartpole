#pragma once

#include <map>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <eigen3/Eigen/Dense>

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

#include "CartPole.cuh"
#include "MCMPC_config.cuh"

class MCMPC_CartPole
{
public:
    MCMPC_CartPole();
    ~MCMPC_CartPole();
    Eigen::VectorXf solve(const Eigen::VectorXf &target_state, const Eigen::VectorXf &current_state);

private:
    // mpc
    //static const int NX = 4; // state = x angle v angleVel
    //static const int NU = 1; // input = v

    Eigen::MatrixXf R;  // input weights
    Eigen::MatrixXf Rd; // input difference weights
    Eigen::MatrixXf Q;  // state weights
    Eigen::MatrixXf Qf; // final state weights
    // tranfer value
    float input_weight, input_diff_weight, pos_weight, angle_weight, v_weight, angleVelocity_weight;

    // parallel computing variable
    thrust::device_vector<Eigen::MatrixXf> u_array_device;
    thrust::host_vector<Eigen::MatrixXf> u_array_host;
    thrust::device_vector<Eigen::VectorXf> input_list_device;
    thrust::host_vector<Eigen::VectorXf> input_list_host;
    thrust::device_vector<float> cost_array_device;
    thrust::host_vector<float> cost_array_host;
    thrust::device_vector<Eigen::VectorXf> x_array;

    curandState *random_seed;

    // GPU config
    unsigned int THREAD_PER_BLOCK;
    unsigned int NUM_BLOCKS;

    const float COST_OVER_VALUE = 10000;

    // constrait

    //mcmpc config
    MCMPC_config config;

};

// cuda device constant
__constant__ int d_NX, d_NU;
__constant__ int d_HORIZONS, d_INPUT_THREADS_NUM, d_TOP_INPUTS_NUM, d_DT;
__constant__ float d_ITERATION_TH, d_max_INPUT, d_X_MAX, d_COST_OVER_VALUE;
__constant__ MCMPC_config d_config;

/*
__constant__ Eigen::VectorXf d_R, d_Rd;
__constant__ Eigen::VectorXf d_Q, d_Qf;
__constant__ float d_LOWER_INPUT[NU] = {-max_INPUT};
__constant__ float d_UPPER_INPUT[NU] = {max_INPUT};
__constant__ float d_LOWER_STATE[NX] = {-X_MAX, -INFINITY, -INFINITY, -INFINITY};
__constant__ float d_UPPER_STATE[NX] = {X_MAX, INFINITY, INFINITY, INFINITY};
*/


unsigned int CountBlocks(unsigned int thread_num, unsigned int thread_per_block);
// cuda functions
__global__ void ParallelMonteCarloSimulation(Eigen::MatrixXf *u_array, float *cost_array, const Eigen::VectorXf &_target_state, const Eigen::VectorXf &_current_state, Eigen::VectorXf *x_array, Eigen::VectorXf *mean, curandState *random_seed);

__device__ float input_constrain(const float _input, const float lower_bound, const float upper_bound);
__device__ float state_constrain(const float _state, const float lower_bound, const float upper_bound);
__device__ inline float GenerateRadomInput(curandState *random_seed, unsigned int id, float mean, float variance);