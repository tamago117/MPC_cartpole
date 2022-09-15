#include <iostream>


#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/sequence.h>

#include "MCMPC_CartPole.cuh"

unsigned int CountBlocks(unsigned int thread_num, unsigned int thread_per_block)
{
    unsigned int num;
    num = thread_num / thread_per_block;
    if (thread_num < thread_per_block || thread_num % thread_per_block > 0)
        num++;
    return num;
}

__device__ inline float GenerateRadomInput(curandState *random_seed, unsigned int id, float mean, float variance)
{
    float ret_value;
    curandState local_seed;
    local_seed = random_seed[id];
    ret_value = curand_normal(&local_seed) * variance + mean;
    return ret_value;
}

__global__ void ParallelMonteCarloSimulation(Eigen::MatrixXf *u_array, float *cost_array, const Eigen::VectorXf &_target_state, const Eigen::VectorXf &_current_state, Eigen::VectorXf *x_array, Eigen::VectorXf *mean, curandState *random_seed)
{
    // prepare random input threads
    int id = blockDim.x * blockIdx.x + threadIdx.x;

    unsigned int seq = id;


    Eigen::VectorXf current_state = _current_state;
    Eigen::VectorXf xref = _target_state;

    x_array[0] = current_state;

    for (int horizon = 0; horizon < d_HORIZONS-1; ++horizon)
    {
        for(int input_dim = 0; input_dim < d_NU; ++input_dim)
        {
            // TODO : determine id
            unsigned int shifted_seq_id = seq + input_dim *(d_HORIZONS * d_NU);
            u_array[id](horizon,input_dim) = GenerateRadomInput(random_seed, shifted_seq_id, mean[horizon+1](input_dim), d_config.INPUT_DEV[input_dim]);

            // input constrain
            u_array[id](horizon,input_dim) = input_constrain(u_array[id](horizon,input_dim), d_config.LOWER_INPUT[input_dim], d_config.UPPER_INPUT[input_dim]);
        }

        // input -> state
        x_array[horizon + 1] = CartPole::dynamics(x_array[horizon], u_array[id].row(horizon), d_DT);

        seq += d_INPUT_THREADS_NUM;
    }

    // last input
    u_array[id].row(d_HORIZONS - 1) = u_array[id].row(d_HORIZONS - 2);
    // last : input -> state
    x_array[d_HORIZONS] = CartPole::dynamics(x_array[d_HORIZONS - 1], u_array[id].row(d_HORIZONS - 1), d_DT);

    ///////////////////// evaluation ///////////////
    // stage cost
    for (int horizon = 0; horizon < d_HORIZONS; ++horizon){
        // state diff
        for(int state_dim = 0; state_dim < d_NX; ++state_dim){
            cost_array[id] += d_config.Q[state_dim] * pow(x_array[horizon](state_dim) - xref(state_dim), 2);
        }
        // input
        for(int input_dim = 0; input_dim < d_NU; ++input_dim){
            cost_array[id] += d_config.R[input_dim] * pow(u_array[id](horizon,input_dim), 2);
        }
    }
    // final state cost
    for(int state_dim = 0; state_dim < d_NX; ++state_dim){
        cost_array[id] += d_config.Qf[state_dim] * pow(x_array[d_HORIZONS](state_dim) - xref(state_dim), 2);
    }

    ///////////////////////////////////////////////

    /////////////////// state constrain /////////////
    for (int horizon = 0; horizon < d_HORIZONS; ++horizon){
        for(int state_dim = 0; state_dim < d_NX; ++state_dim){
            cost_array[id] += state_constrain(x_array[horizon](state_dim), d_config.LOWER_STATE[state_dim], d_config.UPPER_STATE[state_dim]);
        }
    }
    ///////////////////////////////////////////////
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
    /*
    input_weight = 0.01;
    input_diff_weight = 0.01;
    pos_weight = 2.0;
    angle_weight = 4.0;
    v_weight = 0.01;
    angleVelocity_weight = 0.01;

    R = Eigen::MatrixXf(1, NU);
    R << input_weight;
    Rd = Eigen::MatrixXf(1, NU);
    Rd << input_diff_weight;
    Q = Eigen::MatrixXf(1, NX);
    Q << pos_weight, angle_weight, v_weight, angleVelocity_weight;
    Qf = Eigen::MatrixXf(1, NX);
    Qf << pos_weight, angle_weight, v_weight, angleVelocity_weight;
    */

    //gpu config
    THREAD_PER_BLOCK = 256;
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
    INPUT_DEV = Eigen::VectorXf(NU);
    INPUT_DEV << max_INPUT / 2;

    LOWER_INPUT = Eigen::VectorXf(NU);
    UPPER_INPUT = Eigen::VectorXf(NU);
    LOWER_INPUT << -max_INPUT;
    UPPER_INPUT << max_INPUT;

    LOWER_STATE = Eigen::VectorXf(NX);
    UPPER_STATE = Eigen::VectorXf(NX);
    LOWER_STATE << -X_MAX, -INFINITY, -INFINITY, -INFINITY;
    UPPER_STATE << max_INPUT, INFINITY, INFINITY, INFINITY;
    */

    //thrust initialize
    thrust::device_vector<Eigen::MatrixXf> u_array_device(INPUT_THREADS_NUM, Eigen::VectorXf(HORIZONS, NU));
    thrust::host_vector<Eigen::MatrixXf> u_array_host(INPUT_THREADS_NUM, Eigen::VectorXf(HORIZONS, NU));
    thrust::device_vector<Eigen::VectorXf> input_list_device(HORIZONS, Eigen::VectorXf(NU));
    thrust::host_vector<Eigen::VectorXf> input_list_host(HORIZONS, Eigen::VectorXf(NU));
    thrust::device_vector<Eigen::VectorXf> x_array(HORIZONS + 1, Eigen::VectorXf(NX));
    thrust::device_vector<float> cost_array_device(INPUT_THREADS_NUM);
    thrust::host_vector<float> cost_array_host(INPUT_THREADS_NUM);
    
    

    //curand initialize
    float num_random_seed = INPUT_THREADS_NUM * (NU + 1) * HORIZONS;
    cudaMalloc((void**)&random_seed, num_random_seed * sizeof(curandState));

    //cuda device constant initialize
    cudaMemcpyToSymbol(d_NX, &NX, sizeof(int));
    cudaMemcpyToSymbol(d_NU, &NU, sizeof(int));
    cudaMemcpyToSymbol(d_HORIZONS, &HORIZONS, sizeof(int));
    cudaMemcpyToSymbol(d_INPUT_THREADS_NUM, &INPUT_THREADS_NUM, sizeof(int));
    cudaMemcpyToSymbol(d_TOP_INPUTS_NUM, &TOP_INPUTS_NUM, sizeof(int));
    cudaMemcpyToSymbol(d_DT, &DT, sizeof(float));
    cudaMemcpyToSymbol(d_ITERATION_TH, &ITERATION_TH, sizeof(float));
    cudaMemcpyToSymbol(d_max_INPUT, &max_INPUT, sizeof(float));
    cudaMemcpyToSymbol(d_X_MAX, &X_MAX, sizeof(float));
    cudaMemcpyToSymbol(d_COST_OVER_VALUE, &COST_OVER_VALUE, sizeof(float));
    cudaMemcpyToSymbol(d_config, &config, sizeof(MCMPC_config));
}

MCMPC_CartPole::~MCMPC_CartPole()
{
    cudaFree(random_seed);
}

Eigen::VectorXf MCMPC_CartPole::solve(const Eigen::VectorXf &target_state, const Eigen::VectorXf &current_state)
{
    Eigen::VectorXf input_result;
    thrust::host_vector<Eigen::VectorXf> pre_input_list;

    // repeat calculations for output accuracy
    for (int i = 0; i < ITERATIONS; ++i)
    {
        pre_input_list = input_list_host;

        thrust::copy(input_list_host.begin(), input_list_host.end(), input_list_device.begin());
        ParallelMonteCarloSimulation<<<NUM_BLOCKS, THREAD_PER_BLOCK>>>(thrust::raw_pointer_cast(u_array_device.data()), thrust::raw_pointer_cast(cost_array_device.data()), target_state, current_state, thrust::raw_pointer_cast(x_array.data()), thrust::raw_pointer_cast(input_list_device.data()), random_seed);

        CHECK(cudaDeviceSynchronize());

        // sort by thrust
        //thrust::sequence(sample_index.begin(), sample_index.end());
        //thrust::sort_by_key(cost_array.begin(), cost_array.end(), sample_index.begin());
        thrust::sort_by_key(cost_array_device.begin(), cost_array_device.end(), u_array_device.begin());
        CHECK(cudaDeviceSynchronize());
        thrust::sort(cost_array_device.begin(), cost_array_device.end());
        CHECK(cudaDeviceSynchronize());

        //GetWeight<<<NUM_BLOCKS, THREAD_PER_BLOCK>>>();
        //calculate sum of weight by thrust
        float lam = thrust::reduce(cost_array_device.begin(), cost_array_device.begin()+TOP_INPUTS_NUM, 0.0);
        thrust::copy(u_array_device.begin(), u_array_device.end(), u_array_host.begin());
        thrust::copy(cost_array_device.begin(), cost_array_device.end(), cost_array_host.begin());

        CHECK(cudaDeviceSynchronize());

        float denom = 0;
        for (int i = 0; i < TOP_INPUTS_NUM; i++)
        {
            // denom += std::exp(-cost[INPUT_THREADS_NUM-1 - i]/lam);
            denom += std::exp(-cost_array_host[i] / lam);
        }
        for (int horizon = 0; horizon < HORIZONS; ++horizon)
        {
            Eigen::VectorXf molecule(NU);
            for (int i = 0; i < TOP_INPUTS_NUM; i++)
            {
                // molecule += std::exp(-cost[INPUT_THREADS_NUM-1 - i]/lam) * u[INPUT_THREADS_NUM-1 - i][horizon];
                molecule += std::exp(-cost_array_host[i] / lam) * u_array_host[i].row(horizon);
            }
            Eigen::VectorXf result = molecule / denom;

            input_list_host[horizon] = result;
        }

        std::cout << "min cost : " << cost_array_host.front() << std::endl;
        input_result = input_list_host.front();



        float du = 0;
        for (int i = 0; i < HORIZONS; ++i)
        {
            du += (pre_input_list[i] - input_list_host[i]).cwiseAbs().sum();
        }
        if (du < ITERATION_TH)
        {
            break;
        }
    }

    return input_result;
}