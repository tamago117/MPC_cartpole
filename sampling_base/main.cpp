#include <iostream>
#include <eigen3/Eigen/Dense>
#include "include/matplotlibcpp.h"
#include "include/MCMPC_CartPole.hpp"

const double SIM_TIME = 500.0;

const double HORIZONS = 30;
const double max_INPUT = 100;
const double DT = 0.02;

//cart pole parameter
const double CART_M = 2.0;
const double POLE_M = 0.2;
const double POLE_L = 0.5;
const double CART_W = 0.25;
const double CART_H = 0.12;
const double WHEEL_RADIUS = 0.02;

//state weight
const double W_X  = 5.0;
const double W_ANGLE = 1.0;
const double W_V = 1.0;
//input weight
const double W_INPUT = 0.01;
const double W_INPUT_D = 0.1;

const double FINISH_ETHETA = 0.1;

void plot_cart(double x, double theta)
{
    std::vector<double> cx{-CART_W / 2.0, CART_W / 2.0, CART_W /2.0,
                           -CART_W / 2.0, -CART_W / 2.0};
    std::vector<double> cy{0.0, 0.0, CART_H, CART_H, 0.0};
    /*for(auto& pos : cy){
        pos += WHEEL_RADIUS*2.0;
    }*/

    for(auto& pos : cx){
        pos += x;
    }

    std::vector<double> bx{0.0, POLE_L * sin(-theta+M_PI)};
    for(auto& pos : bx){
        pos += x;
    }
    std::vector<double> by{CART_H, POLE_L*cos(-theta+M_PI) + CART_H};
    /*for(auto& pos : by){
        pos += WHEEL_RADIUS*2.0;
    }*/

    const int radians_num = 100;
    std::vector<double> ox(radians_num+1);
    std::vector<double> oy(radians_num+1);
    for(int i=0; i<radians_num+1; ++i){
        double radian = (2*M_PI/radians_num)*i;
        ox[i] = WHEEL_RADIUS*cos(radian);
        oy[i] = WHEEL_RADIUS*sin(radian);
    }

    std::vector<double> rwx(ox.size());
    std::vector<double> rwy(ox.size());
    std::vector<double> lwx(ox.size());
    std::vector<double> lwy(ox.size());
    std::vector<double> wx(ox.size());
    std::vector<double> wy(ox.size());
    for(int i=0; i<rwx.size(); ++i){
        //rwx[i] = ox[i] + CART_W / 4.0 + x;
        //rwy[i] = oy[i] + WHEEL_RADIUS;
        //lwx[i] = ox[i] - CART_W / 4.0 + x;
        //lwy[i] = oy[i] + WHEEL_RADIUS;
        wx[i] = ox[i] + bx.back();
        wy[i] = oy[i] + by.back();
    }

    matplotlibcpp::plot(cx, cy, "-b");
    matplotlibcpp::plot(bx, by, "-k");
    matplotlibcpp::plot(rwx, rwy, "-k");
    matplotlibcpp::plot(lwx, lwy, "-k");
    matplotlibcpp::plot(wx, wy, "-k");

}

int main()
{
    double time = 0.0;
    double v, w;

    //target and current state
    Eigen::VectorXd target(4);
    Eigen::VectorXd current(4);
    Eigen::VectorXd input(1);
    target << 1.0, M_PI, 0.0, 0.0;
    current << 0.0, M_PI+0.4, 0.0, 0.0;

    MCMPC_CartPole mcmpc_cartpole;

    while(SIM_TIME >= time)
    {
        time += DT;

        //mpc solve
        input = mcmpc_cartpole.solve(target, current);
        std::cout<<"input : "<<input<<std::endl;

        //store data history
        static CartPole cartpole(CART_M, POLE_M, POLE_L, DT);
        current = cartpole.dynamics(current, input);

        matplotlibcpp::clf();
        plot_cart(current(0), current(1));
        matplotlibcpp::title("MPC cartpole\n time[s]:" + std::to_string(time) + " theta[degree]:" + std::to_string(current(1)*180/3.14) + " velocity[m/s]" + std::to_string(current(2)));
        matplotlibcpp::axis("equal");
        matplotlibcpp::grid(true);
        matplotlibcpp::pause(0.001);

        //finish judge
        /*if(current(1) > 3.14 || current(1) < -3.14){
            break;
        }*/

    }

    return 0;
}