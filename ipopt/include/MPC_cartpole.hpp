#pragma once

#include <iostream>
#include <vector>
#include <map>
#include <eigen3/Eigen/Core>
#define HAVE_CSTDDEF
#include <cppad/ipopt/solve.hpp>
#undef HAVE_CSTDDEF

class MPC_cartpole
{
    public:
        MPC_cartpole();

        // x[x, v, theta, w]  u[F]
        Eigen::VectorXd get_model(Eigen::VectorXd x, Eigen::VectorXd u);

        // Solve the model given an initial state and polynomial coefficients.
        // Return the first actuatotions.
        std::vector<double> Solve(Eigen::VectorXd state, Eigen::VectorXd target);
        void LoadParams(const std::map<std::string, double> &params);
        std::vector<double> mpc_x, mpc_theta;

    private:
        double CART_M, POLE_M;
        double POLE_L;
        const double g = 9.81;

        double DT;

        double max_INPUT, BOUND_VALUE;
        int HORIZONS;
        int X_START, ANGLE_START, V_START, W_START, INPUT_START;
        std::map<std::string, double> params;

};

class FG_eval
{
    public:
        double CART_M, POLE_M;
        double POLE_L;
        const double g = 9.81;

        double DT, REF_X, REF_ANGLE, REF_V;
        double  W_ANGLE, W_X, W_V, W_INPUT, W_INPUT_D;
        int HORIZONS;
        int X_START, ANGLE_START, V_START, W_START, INPUT_START;

        CppAD::AD<double> cost_angle, cost_x, cost_vel;

        // Constructor
        FG_eval(Eigen::VectorXd target)
        {

            // Set default value
            DT = 0.1;  // in sec
            REF_X = target(0);
            REF_ANGLE  = target(2);
            REF_V = target(1);

            W_X = 1;
            W_ANGLE = 1;
            W_V = 1;
            W_INPUT = 0.01;
            W_INPUT_D = 0.01;

            HORIZONS = 30;
            X_START = 0;
            ANGLE_START = X_START + HORIZONS;
            V_START = ANGLE_START + HORIZONS;
            W_START = V_START + HORIZONS;
            INPUT_START = W_START + HORIZONS - 1;
        }

        // Load parameters for constraints
        void LoadParams(const std::map<std::string, double> &params)
        {
            CART_M = params.find("CART_M") != params.end() ? params.at("CART_M") : CART_M;
            POLE_M = params.find("POLE_M") != params.end() ? params.at("POLE_M") : POLE_M;
            POLE_L = params.find("POLE_L") != params.end() ? params.at("POLE_L") : POLE_L;

            DT = params.find("DT") != params.end() ? params.at("DT") : DT;
            HORIZONS = params.find("HORIZONS") != params.end() ? params.at("HORIZONS") : HORIZONS;
            //REF_X  = params.find("REF_X") != params.end() ? params.at("REF_X") : REF_X;
            //REF_ANGLE = params.find("REF_ANGLE") != params.end() ? params.at("REF_ANGLE") : REF_ANGLE;

            W_X  = params.find("W_X") != params.end() ? params.at("W_X") : W_X;
            W_ANGLE = params.find("W_ANGLE") != params.end() ? params.at("W_ANGLE") : W_ANGLE;
            W_V = params.find("W_V") != params.end() ? params.at("W_V") : W_V;
            W_INPUT = params.find("W_INPUT") != params.end() ? params.at("W_INPUT") : W_INPUT;
            W_INPUT_D = params.find("W_INPUT_D") != params.end() ? params.at("W_INPUT_D") : W_INPUT_D;

            X_START = 0;
            ANGLE_START = X_START + HORIZONS;
            V_START = ANGLE_START + HORIZONS;
            W_START = V_START + HORIZONS;
            INPUT_START = W_START + HORIZONS - 1;
            //cout << "\n!! FG_eval Obj parameters updated !! " << HORIZONS << endl;
        }

        // MPC implementation (cost func & constraints)
        typedef CPPAD_TESTVECTOR(CppAD::AD<double>) ADvector;
        // fg: function that evaluates the objective and constraints using the syntax
        void operator()(ADvector& fg, const ADvector& vars)
        {
            // fg[0] for cost function
            fg[0] = 0;
            cost_x = 0;
            cost_angle = 0;
            cost_vel = 0;

            for(int i = 0; i < HORIZONS; ++i){
                fg[0] += W_X * CppAD::pow(vars[X_START + i] - REF_X, 2); // x error
                fg[0] += W_ANGLE * CppAD::pow(vars[ANGLE_START + i] - REF_ANGLE, 2); // angle error
                fg[0] += W_V * CppAD::pow(vars[V_START + i] - REF_V, 2); // velocity error

                cost_x += (W_X * CppAD::pow(vars[X_START + i] - REF_X, 2));
                cost_angle += (W_ANGLE * CppAD::pow(vars[ANGLE_START + i] - REF_ANGLE, 2));
                cost_vel += (W_V * CppAD::pow(vars[V_START + i] - REF_V, 2));
            }
            //std::cout << "-----------------------------------------------" <<std::endl;
            //std::cout << "cost_x : " << cost_x << ", cost_angle : " << cost_angle <<", cost_velocity : "<< cost_vel << std::endl;


            // Minimize the use of actuators.
            for (int i = 0; i < HORIZONS - 1; i++){
                fg[0] += W_INPUT * CppAD::pow(vars[INPUT_START + i], 2);
            }
            //std::cout << "cost of actuators: " << fg[0] << std::endl;

            // Minimize the value gap between sequential actuations.
            for(int i = 0; i < HORIZONS - 2; i++){
                fg[0] += W_INPUT_D * CppAD::pow(vars[INPUT_START + i + 1] - vars[INPUT_START + i], 2);
            }
            //std::cout << "cost of gap: " << fg[0] << std::endl;

            // fg[x] for constraints
            // Initial constraints
            fg[1 + X_START] = vars[X_START];
            fg[1 + ANGLE_START] = vars[ANGLE_START];
            fg[1 + V_START] = vars[V_START];
            fg[1 + W_START] = vars[W_START];

            // Add system dynamic model constraint
            for(int i = 0; i < HORIZONS - 1; i++){
                // The state at time t+1 .
                CppAD::AD<double> x1 = vars[X_START + i + 1];
                CppAD::AD<double> theta1 = vars[ANGLE_START + i + 1];
                CppAD::AD<double> v1 = vars[V_START + i + 1];
                CppAD::AD<double> w1 = vars[W_START + i + 1];


                // The state at time t.
                CppAD::AD<double> x0 = vars[X_START + i];
                CppAD::AD<double> theta0 = vars[ANGLE_START + i];
                CppAD::AD<double> v0 = vars[V_START + i];
                CppAD::AD<double> w0 = vars[W_START + i];

                //std::cout << theta0<<std::endl;

                // Only consider the actuation at time t.
                CppAD::AD<double> input0 = vars[INPUT_START + i];

                // model constraints
                CppAD::AD<double> S = CppAD::sin(theta0);
                CppAD::AD<double> C = CppAD::cos(theta0);
                CppAD::AD<double> alpha = CART_M + POLE_M - POLE_M*CppAD::pow(C, 2);

                fg[2 + X_START + i] = x1 - (x0 + v0 * DT);
                fg[2 + ANGLE_START + i] = theta1 - (theta0 +  w0 * DT);
                fg[2 + V_START + i] = v1 - (v0 + ((1/alpha)*(-POLE_M*POLE_L*CppAD::pow(w0, 2)*S+POLE_M*g*C*S) + (1/alpha)*input0) * DT);
                fg[2 + W_START + i] = w1 - (w0 +  ((1/(alpha*POLE_L))*(-POLE_M*POLE_L*CppAD::pow(w0, 2)*S*C + (CART_M+POLE_M)*g*S) + (1/(alpha*POLE_L))*input0) * DT);
            }
        }

};

MPC_cartpole::MPC_cartpole()
{
    CART_M = 1.0;
    POLE_M = 0.3;
    POLE_L = 2.0;
    DT = 0.1;

    HORIZONS = 20;
    max_INPUT = 1.0;
    BOUND_VALUE = 100;

    X_START = 0;
    ANGLE_START = X_START + HORIZONS;
    V_START = ANGLE_START + HORIZONS;
    W_START = V_START + HORIZONS;
    INPUT_START = W_START + HORIZONS - 1;
}

Eigen::VectorXd MPC_cartpole::get_model(Eigen::VectorXd x, Eigen::VectorXd u)
{
    Eigen::VectorXd A(4);
    Eigen::VectorXd B(4);

    double F = u(0);

    double S = sin(x(2));
    double C = cos(x(2));

    double alpha = CART_M + POLE_M - POLE_M*pow(C, 2);
    A << x(1),
         (1/alpha)*(-POLE_M*POLE_L*pow(x(3), 2)*S+POLE_M*g*C*S),
         x(3),
         (1/(alpha*POLE_L))*(-POLE_M*POLE_L*pow(x(3), 2)*S*C + (CART_M+POLE_M)*g*S);

    A = x + A*DT;

    B << 0,
         (1/alpha)*u(0),
         0,
         (1/(alpha*POLE_L))*u(0);

    B = B*DT;

    /*double alpha = 2*(CART_M + POLE_M) - POLE_M*pow(C, 2);
    A << x(1),
         (1/alpha)*(-2*POLE_M*POLE_L*pow(x(3), 2)*S+POLE_M*g*C*S),
         x(3),
         (1/(alpha*POLE_L))*(-POLE_M*POLE_L*pow(x(3), 2)*S*C + (CART_M+POLE_M)*g*S);

    A = x + A*DT;

    B << 0,
         (2/alpha)*F,
         0,
         (1/(alpha*POLE_L))*F;

    B = B*DT;*/

    return A+B;
}

void MPC_cartpole::LoadParams(const std::map<std::string, double> &params_)
{
    params = params_;

    //Init parameters for MPC object
    CART_M = params.find("CART_M") != params.end() ? params.at("CART_M") : CART_M;
    POLE_M = params.find("POLE_M") != params.end() ? params.at("POLE_M") : POLE_M;
    POLE_L = params.find("POLE_L") != params.end() ? params.at("POLE_L") : POLE_L;
    DT = params.find("DT") != params.end() ? params.at("DT") : DT;

    HORIZONS = params.find("HORIZONS") != params.end() ? params.at("HORIZONS") : HORIZONS;
    max_INPUT = params.find("max_INPUT") != params.end() ? params.at("max_INPUT") : max_INPUT;
    BOUND_VALUE = params.find("BOUND_VALUE") != params.end() ? params.at("BOUND_VALUE") : BOUND_VALUE;

    X_START = 0;
    ANGLE_START = X_START + HORIZONS;
    V_START = ANGLE_START + HORIZONS;
    W_START = V_START + HORIZONS;
    INPUT_START = W_START + HORIZONS - 1;

    std::cout << "\n!! MPC Obj parameters updated !! " << std::endl;
}

std::vector<double> MPC_cartpole::Solve(Eigen::VectorXd state, Eigen::VectorXd target)
{
    bool ok = true;
    typedef CPPAD_TESTVECTOR(double) Dvector;
    const double x = state[0];
    const double v = state[1];
    const double theta = state[2];
    const double w = state[3];

    // Set the number of model variables (includes both states and inputs).
    // For example: If the state is a 4 element vector, the actuators is a 2
    // element vector and there are 10 timesteps. The number of variables is:
    size_t n_vars = HORIZONS * 4 + (HORIZONS - 1) * 1;
    
    // Set the number of constraints
    size_t n_constraints = HORIZONS * 4;

    // Initial value of the independent variables.
    // SHOULD BE 0 besides initial state.
    Dvector vars(n_vars);
    for (int i = 0; i < n_vars; i++){
        vars[i] = 0;
    }

    // Set the initial variable values
    vars[X_START] = x;
    vars[ANGLE_START] = theta;
    vars[V_START] = v;
    vars[W_START] = w;

    // Set lower and upper limits for variables.
    Dvector vars_lowerbound(n_vars);
    Dvector vars_upperbound(n_vars);
    
    // Set all non-actuators upper and lowerlimits
    // to the max negative and positive values.
    for (int i = 0; i < INPUT_START; i++){
        vars_lowerbound[i] = -BOUND_VALUE;
        vars_upperbound[i] = BOUND_VALUE;
    }
    // The upper and lower limits of input
    // [N].
    for (int i = INPUT_START; i < n_vars; i++){
        vars_lowerbound[i] = -max_INPUT;
        vars_upperbound[i] = max_INPUT;
    }


    // Lower and upper limits for the constraints
    // Should be 0 besides initial state.
    Dvector constraints_lowerbound(n_constraints);
    Dvector constraints_upperbound(n_constraints);
    for (int i = 0; i < n_constraints; i++){
        constraints_lowerbound[i] = 0;
        constraints_upperbound[i] = 0;
    }
    
    constraints_lowerbound[X_START] = x;
    constraints_lowerbound[ANGLE_START] = theta;
    constraints_lowerbound[V_START] = v;
    constraints_lowerbound[W_START] = w;
    constraints_upperbound[X_START] = x;
    constraints_upperbound[ANGLE_START] = theta;
    constraints_upperbound[V_START] = v;
    constraints_upperbound[W_START] = w;
    
    // object that computes objective and constraints
    FG_eval fg_eval(target);
    fg_eval.LoadParams(params);


    // options for IPOPT solver
    std::string options;
    // Uncomment this if you'd like more print information
    options += "Integer print_level  0\n";
    // NOTE: Setting sparse to true allows the solver to take advantage
    // of sparse routines, this makes the computation MUCH FASTER. If you
    // can uncomment 1 of these and see if it makes a difference or not but
    // if you uncomment both the computation time should go up in orders of
    // magnitude.
    options += "Sparse  true        forward\n";
    options += "Sparse  true        reverse\n";
    // NOTE: Currently the solver has a maximum time limit of 0.5 seconds.
    // Change this as you see fit.
    options += "Numeric max_cpu_time          0.5\n";

    // place to return solution
    CppAD::ipopt::solve_result<Dvector> solution;

    // solve the problem
    CppAD::ipopt::solve<Dvector, FG_eval>(
      options, vars, vars_lowerbound, vars_upperbound, constraints_lowerbound,
      constraints_upperbound, fg_eval, solution);

    // Check some of the solution values
    ok &= solution.status == CppAD::ipopt::solve_result<Dvector>::success;

    // Cost
    auto cost = solution.obj_value;
    std::cout << "------------ Total Cost(solution): " << cost << "------------" << std::endl;

    this->mpc_x = {};
    this->mpc_theta = {};
    for (int i = 0; i < HORIZONS; i++){
        this->mpc_x.push_back(solution.x[X_START + i]);
        this->mpc_theta.push_back(solution.x[ANGLE_START + i]);
    }
    std::vector<double> result;
    result.push_back(solution.x[INPUT_START]);
    return result;
}