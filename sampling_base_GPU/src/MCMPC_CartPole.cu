#include <iostream>


#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "MCMPC_CartPole.cuh"
#include "MCMPC_config.cuh"

// cuda device constant
__constant__ int d_NX, d_NU;
__constant__ int d_HORIZONS, d_INPUT_THREADS_NUM, d_TOP_INPUTS_NUM;
__constant__ float d_DT, d_ITERATION_TH, d_max_INPUT, d_X_MAX, d_COST_OVER_VALUE;
__constant__ MCMPC_config d_config;

__device__ float GenerateRadomInput(curandState *random_seed, unsigned int id, float mean, float variance)
{
    float ret_value;
    curandState local_seed;
    local_seed = random_seed[id];
    ret_value = curand_normal(&local_seed) * variance + mean;
    return ret_value;
}

__global__ void ParallelMonteCarloSimulation(u_array *u_array, float *cost_array, const vectorF<NX> xref, const vectorF<NX> current_state, x_array *x_array, vectorF<NU> *mean, curandState *random_seed)
{
    // prepare random input threads
    int id = blockDim.x * blockIdx.x + threadIdx.x;
    if(id < d_INPUT_THREADS_NUM){
        unsigned int seq = id;

        //initial state
        x_array[id].x[0] = current_state;
        for (int horizon = 0; horizon < d_HORIZONS-1; ++horizon){
            for(int input_dim = 0; input_dim < d_NU; ++input_dim){
                // TODO : determine id
                unsigned int shifted_seq_id = seq + input_dim *(d_HORIZONS * d_NU);
                u_array[id].u[horizon].vector[input_dim] = GenerateRadomInput(random_seed, shifted_seq_id, mean[horizon+1].vector[input_dim], d_config.INPUT_DEV[input_dim]);
                // input constrain
                u_array[id].u[horizon].vector[input_dim] = input_constrain(u_array[id].u[horizon].vector[input_dim], d_config.LOWER_INPUT[input_dim], d_config.UPPER_INPUT[input_dim]);
            }

            // input -> state
            x_array[id].x[horizon + 1] = CartPole::dynamics(x_array[id].x[horizon], u_array[id].u[horizon], d_DT);

            seq += d_INPUT_THREADS_NUM;
        }

        // last input
        u_array[id].u[d_HORIZONS - 1] = u_array[id].u[d_HORIZONS - 2];
        // last : input -> state
        x_array[id].x[d_HORIZONS] = CartPole::dynamics(x_array[id].x[d_HORIZONS - 1], u_array[id].u[d_HORIZONS - 1], d_DT);

        ///////////////////// evaluation ///////////////
        // stage cost
        for (int horizon = 0; horizon < d_HORIZONS; ++horizon){
            // state diff
            for(int state_dim = 0; state_dim < d_NX; ++state_dim){
                //cost_array[id] += d_config.Q[state_dim] * pow(x_array[id].x[horizon].vector[state_dim] - xref.vector[state_dim], 2);
                cost_array[id] += d_config.Q[state_dim] * abs(x_array[id].x[horizon].vector[state_dim] - xref.vector[state_dim]);
            }
            // input
            for(int input_dim = 0; input_dim < d_NU; ++input_dim){
                //cost_array[id] += d_config.R[input_dim] * pow(u_array[id].u[horizon].vector[input_dim], 2);
                cost_array[id] += d_config.R[input_dim] * abs(u_array[id].u[horizon].vector[input_dim]);
            }
        }
        // final state cost
        for(int state_dim = 0; state_dim < d_NX; ++state_dim){
            //cost_array[id] += d_config.Qf[state_dim] * pow(x_array[id].x[d_HORIZONS].vector[state_dim] - xref.vector[state_dim], 2);
            cost_array[id] += d_config.Qf[state_dim] * abs(x_array[id].x[d_HORIZONS].vector[state_dim] - xref.vector[state_dim]);
        }

        ///////////////////////////////////////////////

        /////////////////// state constrain /////////////
        for (int horizon = 0; horizon < d_HORIZONS+1; ++horizon){
            for(int state_dim = 0; state_dim < d_NX; ++state_dim){
                cost_array[id] += state_constrain(x_array[id].x[horizon].vector[state_dim], d_config.LOWER_STATE[state_dim], d_config.UPPER_STATE[state_dim]);
            }
        }
        ///////////////////////////////////////////////
    }
}

__device__ float input_constrain(const float _input, const float lower_bound, const float upper_bound)
{
    if (_input < lower_bound)
    {
        return lower_bound;
    }
    else if (_input > upper_bound)
    {
        return upper_bound;
    }

    return _input;
}

__device__ float state_constrain(const float _state, const float lower_bound, const float upper_bound)
{
    if (_state < lower_bound)
    {
        return d_COST_OVER_VALUE;
    }
    else if (_state > upper_bound)
    {
        return d_COST_OVER_VALUE;
    }

    return 0;
}


MCMPC_CartPole::MCMPC_CartPole()
{

    //gpu config
    NUM_BLOCKS = CountBlocks(INPUT_THREADS_NUM, THREAD_PER_BLOCK);

    //mcmpc
    /*
    HORIZONS = 100;
    ITERATIONS = 2;
    ITERATION_TH = 0.08;
    INPUT_THREADS_NUM = 1000;
    TOP_INPUTS_NUM = 50;
    max_INPUT = 25;
    X_MAX = 0.5;
    DT = 0.01;
    */

    //thrust initialize
    thrust::device_vector<vectorF<NU>> input_list_device_(HORIZONS);
    thrust::host_vector<vectorF<NU>> input_list_host_(HORIZONS);
    thrust::device_vector<u_array> u_array_device_(INPUT_THREADS_NUM);
    thrust::host_vector<u_array> u_array_host_(INPUT_THREADS_NUM);
    thrust::device_vector<x_array> x_array_device_(INPUT_THREADS_NUM);
    thrust::device_vector<float> cost_array_device_(INPUT_THREADS_NUM);
    thrust::host_vector<float> cost_array_host_(INPUT_THREADS_NUM);
    thrust::device_vector<int> indices_device_(INPUT_THREADS_NUM);
    thrust::host_vector<int> indices_host_(INPUT_THREADS_NUM);

    input_list_device = input_list_device_;
    input_list_host = input_list_host_;
    u_array_device = u_array_device_;
    u_array_host = u_array_host_;
    x_array_device = x_array_device_;
    cost_array_device = cost_array_device_;
    cost_array_host = cost_array_host_;
    indices_device = indices_device_;
    indices_host = indices_host_;

    //curand initialize
    float num_random_seed = INPUT_THREADS_NUM * (NU + 1) * HORIZONS;
    cudaMalloc((void**)&random_seed, num_random_seed * sizeof(curandState));
    SetRandomSeed<<<INPUT_THREADS_NUM, (NU + 1) * HORIZONS>>>(random_seed, rand());
    CHECK(cudaDeviceSynchronize());

    //cuda device constant initialize
    cudaMemcpyToSymbol(d_NX, &NX, sizeof(int));
    cudaMemcpyToSymbol(d_NU, &NU, sizeof(int));
    cudaMemcpyToSymbol(d_HORIZONS, &HORIZONS, sizeof(int));
    cudaMemcpyToSymbol(d_INPUT_THREADS_NUM, &INPUT_THREADS_NUM, sizeof(int));
    cudaMemcpyToSymbol(d_TOP_INPUTS_NUM, &TOP_INPUTS_NUM, sizeof(int));
    cudaMemcpyToSymbol(d_DT, &DT, sizeof(float));
    cudaMemcpyToSymbol(d_max_INPUT, &max_INPUT, sizeof(float));
    cudaMemcpyToSymbol(d_X_MAX, &X_MAX, sizeof(float));
    cudaMemcpyToSymbol(d_COST_OVER_VALUE, &COST_OVER_VALUE, sizeof(float));
    cudaMemcpyToSymbol(d_config, &config, sizeof(MCMPC_config));
}

MCMPC_CartPole::~MCMPC_CartPole()
{
    cudaFree(random_seed);
}

vectorF<NU> MCMPC_CartPole::solve(const vectorF<NX> &target_state, const vectorF<NX> &current_state)
{
    //initialize
    vectorF<NU> input_result;
    thrust::host_vector<vectorF<NU>> pre_input_list;
    float pre_cost = INFINITY;


    // repeat calculations for output accuracy
    for (int i = 0; i < ITERATIONS; ++i)
    {
        thrust::device_vector<float> cost_array_device_(INPUT_THREADS_NUM);
        cost_array_device = cost_array_device_;

        pre_input_list = input_list_host;
        thrust::copy(input_list_host.begin(), input_list_host.end(), input_list_device.begin());
        ParallelMonteCarloSimulation<<<NUM_BLOCKS, THREAD_PER_BLOCK>>>(thrust::raw_pointer_cast(u_array_device.data()), thrust::raw_pointer_cast(cost_array_device.data()), target_state, current_state, thrust::raw_pointer_cast(x_array_device.data()), thrust::raw_pointer_cast(input_list_device.data()), random_seed);

        CHECK(cudaDeviceSynchronize());

        thrust::sequence(indices_device.begin(), indices_device.end());
        thrust::sort_by_key(cost_array_device.begin(), cost_array_device.end(), indices_device.begin());
        CHECK(cudaDeviceSynchronize());

        //calculate sum of weight by thrust
        float lam = thrust::reduce(cost_array_device.begin(), cost_array_device.begin()+TOP_INPUTS_NUM, 0.0);

        thrust::copy(u_array_device.begin(), u_array_device.end(), u_array_host.begin());
        thrust::copy(cost_array_device.begin(), cost_array_device.end(), cost_array_host.begin());
        thrust::copy(indices_device.begin(), indices_device.end(), indices_host.begin());
        CHECK(cudaDeviceSynchronize());

        float denom = 0;
        for (int j = 0; j < TOP_INPUTS_NUM; ++j){
            denom += std::exp(-cost_array_host[j] / lam);
        }
        for (int horizon = 0; horizon < HORIZONS; ++horizon){
            vectorF<NU> molecule;
            //variable initialize
            for (int input_dim = 0; input_dim < NU; input_dim++){
                molecule.vector[input_dim] = 0;
            }

            molecule.vector[0] = 0;
            for (int j = 0; j < TOP_INPUTS_NUM; ++j){
                int array_index = indices_host[j];
                for(int input_dim = 0; input_dim < NU; input_dim++){
                    molecule.vector[input_dim] += std::exp(-cost_array_host[j]/lam) * u_array_host[array_index].u[horizon].vector[input_dim];
                }
            }
            vectorF<NU> result;
            for(int input_dim = 0; input_dim < NU; ++input_dim){
                result.vector[input_dim] = molecule.vector[input_dim] / denom;
            }
            input_list_host[horizon] = result;
        }

        float min_cost = cost_array_host.front();


        // iteration check
        if(min_cost<pre_cost){
            pre_cost = min_cost;
            std::cout << "min cost : " << min_cost << std::endl;
            input_result = input_list_host[0];
        }else{
            input_list_host = pre_input_list;
            input_result = input_list_host[0];
        }
    }

    return input_result;
}