# MPC_cartpole
cart pole by model predictive control

```
#casadi
pip install casadi

#acados
git clone https://github.com/acados/acados
cd acados
git submodule update --recursive --init
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j4
sudo make install -j
cd ..
pip3 install -e interfaces/acados_template
echo export LD_LIBRARY_PATH=$PWD/lib:$LD_LIBRARY_PATH  >> ~/.bashrc
echo export ACADOS_SOURCE_DIR=$PWD >> ~/.bashrc
source ~/.bashrc
```

this program matplotlib-cpp
- [lava/matplotlib-cpp: Extremely simple yet powerful header-only C++ plotting library built on the popular matplotlib](https://github.com/lava/matplotlib-cpp "lava/matplotlib-cpp: Extremely simple yet powerful header-only C++ plotting library built on the popular matplotlib")

and reference matplotlib-cpp-starter
- [AtsushiSakai/matplotlib-cpp-starter: This is a starter kit with matplotlib-cpp](https://github.com/AtsushiSakai/matplotlib-cpp-starter "AtsushiSakai/matplotlib-cpp-starter: This is a starter kit with matplotlib-cpp")
