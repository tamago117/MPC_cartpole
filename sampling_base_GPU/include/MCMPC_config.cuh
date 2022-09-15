#pragma once

const int NX = 4; // state = x angle v angleVel
const int NU = 1; // input = v

// mcmpc
const int HORIZONS = 100;          // horizon length
const int ITERATIONS = 2;        // iteration number
const float ITERATION_TH = 0.08;   // convergence value
const int INPUT_THREADS_NUM = 1000; // mcmpc : input threads(500~(5000))
const int TOP_INPUTS_NUM = 50;        // mcmpc : cost lower inputs
const float X_MAX = 0.5;
const float max_INPUT = 25;
const float DT = 0.01;
const float COST_OVER_VALUE = 10000;

class MCMPC_config
{
public:
    float R[NU] = {0.1}; // input cost
    float Rd[NU] = {0.1}; // input rate cost
    float Q[NX] = {1, 1, 0.1, 0.1}; // state cost
    float Qf[NX] = {1, 1, 0.1, 0.1}; // terminal state cost
    float INPUT_DEV[NU] = {max_INPUT}; // input deviation
    float LOWER_INPUT[NU] = {-max_INPUT}; // lower input
    float UPPER_INPUT[NU] = {max_INPUT}; // upper input
    float LOWER_STATE[NX] = {-X_MAX, -INFINITY, -INFINITY, -INFINITY};
    float UPPER_STATE[NX] = {X_MAX, INFINITY, INFINITY, INFINITY};
};
