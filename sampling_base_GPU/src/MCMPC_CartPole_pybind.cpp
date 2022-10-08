#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>

#include "MCMPC_CartPole.cuh"
#include "MCMPC_config.cuh"

namespace py = pybind11;

class MCMPC_CartPole_pybind
{

public:
    MCMPC_CartPole_pybind()
    {

    }

    py::array_t<float> solve(const py::array_t<float> current_, const py::array_t<float> target_)
    {
        // change variable type
        vectorF<NX> target;
        vectorF<NX> current;
        for (int i = 0; i < NX; i++){
            target.vector[i] = target_.at(i);
            current.vector[i] = current_.at(i);
        }

        vectorF<NU> input_ = mcmpc_cartpole.solve(target, current);

        // change variable type
        py::array_t<float> input = py::array_t<float>(NU);
        for (int i = 0; i < NU; i++){
            input.mutable_at(i) = input_.vector[i];
        }

        return input;
    }

private:
    MCMPC_CartPole mcmpc_cartpole;
};


PYBIND11_MODULE(mcmpc_CartPole, m)
{
    py::class_<MCMPC_CartPole_pybind>(m, "mcmpc_CartPole")
        .def(py::init<>())
        .def("solve", &MCMPC_CartPole_pybind::solve);
}