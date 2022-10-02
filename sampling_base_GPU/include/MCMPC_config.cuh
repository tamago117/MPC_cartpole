#pragma once

const int NX = 4; // state = x angle v angleVel
const int NU = 1; // input = v

// mcmpc
const int HORIZONS = 40;          // horizon length
const int ITERATIONS = 3;        // iteration number
const float ITERATION_TH = 0.5;   // convergence value
const int INPUT_THREADS_NUM = 5000; // mcmpc : input threads(500~(5000))
const int TOP_INPUTS_NUM = INPUT_THREADS_NUM/100;        // mcmpc : cost lower inputs
const float X_MAX = 0.3;
const float max_INPUT = 25;
const float DT = 0.02;
const float COST_OVER_VALUE = 100;

const int THREAD_PER_BLOCK = 256;

class MCMPC_config
{
public:
    float R[NU] = {0.01}; // input cost
    float Rd[NU] = {0.01}; // input rate cost
    float Q[NX] = {1, 2, 0.05, 0.05}; // state cost
    float Qf[NX] = {1, 2, 0.1, 0.1}; // terminal state cost
    float INPUT_DEV[NU] = {max_INPUT}; // input deviation
    float LOWER_INPUT[NU] = {-max_INPUT}; // lower input
    float UPPER_INPUT[NU] = {max_INPUT}; // upper input
    float LOWER_STATE[NX] = {-X_MAX, -INFINITY, -INFINITY, -INFINITY};
    float UPPER_STATE[NX] = {X_MAX, INFINITY, INFINITY, INFINITY};
};

template <int N> class vectorF
{
public:
    float vector[N];
};

class u_array
{
public:
    vectorF<NU> u[HORIZONS];
};

class x_array
{
public:
    vectorF<NX> x[HORIZONS+1];
};
