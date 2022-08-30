#pragma once

#include <map>
#include <vector>
#include <string>
#include <random>
#include <algorithm>
#include <eigen3/Eigen/Dense>

#include "CartPole.hpp"

class MCMPC_CartPole
{
    public:
        MCMPC_CartPole();
        Eigen::VectorXd solve(const Eigen::VectorXd& target_state, const Eigen::VectorXd& current_state);

    private:
        //mpc
        const int NX = 4; //state = x angle v angleVel
        const int NU = 1; //input = v

        Eigen::MatrixXd R; //input weights
        Eigen::MatrixXd Rd; //input difference weights
        Eigen::MatrixXd Q; //state weights
        Eigen::MatrixXd Qf; //final state weights
        //tranfer value
        double input_weight, input_diff_weight, pos_weight, angle_weight, v_weight, angleVelocity_weight;

        //mcmpc
        int HORIZONS; //horizon length
        int ITERATIONS; //iteration number
        double ITERATION_TH; //convergence value
        int INPUT_THREAD; //mcmpc : input threads(500~(5000))
        int TOP_INPUTS; //mcmpc : cost lower inputs
        double max_INPUT;
        double DT;
        const double COST_OVER_VALUE = 1000;

        //model parameters
        double CART_MASS;
        double POLE_MASS;
        double POLE_LENGTH;

        Eigen::VectorXd INPUT_DEV;

        //class variables
        std::vector<double> input_list;

        //functions
        double input_constrain(const double _input, const double lower_bound, const double upper_bound);
        Eigen::VectorXd mcmpc_control(const Eigen::VectorXd& _target_state, const Eigen::VectorXd& _current_state);

        inline double gauss_dis(double mu = 0.0, double sig = 1.0)
        {
            std::random_device seed;
            std::mt19937 engine(seed());            // メルセンヌ・ツイスター法
            // std::minstd_rand0 engine(seed());    // 線形合同法
            // std::ranlux24_base engine(seed());   // キャリー付き減算法

            std::normal_distribution<> dist(mu, sig);

            return dist(engine);
        }

        template <class T> void sort2vectors(std::vector<T> &av, std::vector<std::vector<Eigen::VectorXd>> &bv)
        {
            int n = av.size();
            std::vector<T> p(n), av2(n);
            std::vector<std::vector<Eigen::VectorXd>> bv2(n);
            iota(p.begin(), p.end(), 0);
            sort(p.begin(), p.end(), [&](int a, int b) { return av[a] < av[b]; });
            for (int i = 0; i < n; i++) {
                    av2[i] = av[p[i]];
                    bv2[i] = bv[p[i]];
            }
            av = av2;
            bv = bv2;
        }
};

MCMPC_CartPole::MCMPC_CartPole()
{
    input_weight = 0.01;
    input_diff_weight = 0.01;
    pos_weight = 2.5;
    angle_weight = 2.5;
    v_weight = 0.05;
    angleVelocity_weight = 0.05;

    R = Eigen::MatrixXd(1, NU);
    R << input_weight;
    Rd = Eigen::MatrixXd(1, NU);
    Rd << input_diff_weight;
    Q = Eigen::MatrixXd(1, NX);
    Q << pos_weight, angle_weight, v_weight, angleVelocity_weight;

    //default values
    HORIZONS = 30;
    ITERATIONS = 2;
    ITERATION_TH = 0.08;
    INPUT_THREAD = 1000;
    TOP_INPUTS = 50;
    max_INPUT = 25;
    DT = 0.02;

    CART_MASS = 2.0;
    POLE_MASS = 0.2;
    POLE_LENGTH = 0.5;

    INPUT_DEV = Eigen::VectorXd(NU);
    INPUT_DEV << max_INPUT/2;

    input_list.resize(HORIZONS);
}

Eigen::VectorXd MCMPC_CartPole::solve(const Eigen::VectorXd& target_state, const Eigen::VectorXd& current_state)
{
    Eigen::VectorXd input;
    std::vector<double> pre_input_list;

    //repeat calculations for output accuracy
    for(int i=0; i<ITERATIONS; ++i){
        pre_input_list = input_list;
        input = mcmpc_control(target_state, current_state);

        double du = 0;
        for(int i=0; i<HORIZONS; ++i){
            du += abs(pre_input_list[i] - input_list[i]);
        }
        if(du < ITERATION_TH){
            break;
        }
    }

    return input;
}

double MCMPC_CartPole::input_constrain(const double _input, const double lower_bound, const double upper_bound)
{
    if(_input < lower_bound){
        return lower_bound;
    }
    else if(_input > upper_bound){
        return upper_bound;
    }

    return _input;
}

Eigen::VectorXd MCMPC_CartPole::mcmpc_control(const Eigen::VectorXd& _target_state, const Eigen::VectorXd& _current_state)
{
    Eigen::VectorXd u_result = Eigen::VectorXd(NU);

    Eigen::VectorXd current_state = _current_state;
    Eigen::VectorXd target_state = _target_state;

    //variable initialize
    std::vector<std::vector<Eigen::VectorXd>> u(INPUT_THREAD, std::vector<Eigen::VectorXd>(HORIZONS, Eigen::VectorXd(NU)));
    std::vector<Eigen::VectorXd> x(HORIZONS+1, Eigen::VectorXd(NX));
    x.front() = current_state;
    Eigen::VectorXd xref = target_state;
    std::vector<double> cost(INPUT_THREAD);

    //cart pole dynamics
    static CartPole cartpole(CART_MASS, POLE_MASS, POLE_LENGTH, DT);

    //prepare random input threads
    for(int thread = 0; thread < INPUT_THREAD; ++thread){
        for(int horizon = 0; horizon < HORIZONS; ++horizon){
            u[thread][horizon](0) = input_list[horizon] + gauss_dis() * INPUT_DEV(0);

            //input constrain
            u[thread][horizon](0) = input_constrain(u[thread][horizon](0), -max_INPUT, max_INPUT);

            //input -> state
            x[horizon+1] = cartpole.dynamics(x[horizon], u[thread][horizon]);
        }

        //last input
        u[thread].back() = u[thread][HORIZONS-1];
        //last : input -> state
        x[HORIZONS] = cartpole.dynamics(x[HORIZONS-1], u[thread].back());

        //evaluation
        for(int horizon = 0; horizon < HORIZONS+1; ++horizon){
            //state diff
            cost[thread] += (Q * (x[horizon] - xref).cwiseAbs()).sum();
        }
        for(int horizon = 0; horizon < HORIZONS; ++horizon){
            cost[thread] += (R * u[thread][horizon].cwiseAbs()).sum();
        }
        for(int horizon = 0; horizon < HORIZONS-1; ++horizon){
            cost[thread] += (Rd * (u[thread][horizon+1] - u[thread][horizon]).cwiseAbs()).sum();
        }

    }

    sort2vectors(cost, u);

    double lam = 0;
    double denom = 0;

    for(int i=0; i<TOP_INPUTS; i++){
        //lam += cost[INPUT_THREAD-1 - i];
        lam += cost[i];
        //std::cout<<u[i][0](0)<<" ";
    }
    //std::cout<<std::endl;

    for(int i=0; i<TOP_INPUTS; i++){
        //denom += std::exp(-cost[INPUT_THREAD-1 - i]/lam);
        denom += std::exp(-cost[i]/lam);
    }
    for(int horizon=0; horizon<HORIZONS; ++horizon){
        Eigen::VectorXd molecule(NU);
        for(int i=0; i<TOP_INPUTS; i++){
            //molecule += std::exp(-cost[INPUT_THREAD-1 - i]/lam) * u[INPUT_THREAD-1 - i][horizon];
            molecule += std::exp(-cost[i]/lam) * u[i][horizon];
        }
        Eigen::VectorXd result = molecule/denom;

        input_list[horizon] = result(0);

    }

    std::cout<<"min cost : "<<cost.front() << std::endl;

    u_result(0) = input_list.front();

    return u_result;
}