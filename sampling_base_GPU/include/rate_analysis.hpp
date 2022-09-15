#pragma once
#include <iostream>
#include <chrono>

class rate_analysis
{
    public:
        rate_analysis();
        void start_record();
        double get_rate();

    private:
        std::chrono::system_clock::time_point start, end;
        bool isStart;
};

rate_analysis::rate_analysis()
{
    isStart = false;
}

void rate_analysis::start_record()
{
    isStart = true;
    start = std::chrono::system_clock::now();
}

double rate_analysis::get_rate()
{
    if(!isStart){
        std::cout<<"\nrate_analysis : start_record() is not called!\n";
        return 0;
    }
    isStart = false;
    end = std::chrono::system_clock::now();
    double elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(end-start).count();
    double rate = 1000/elapsed;

    return rate;
}