cmake_minimum_required(VERSION 3.8)
project(sampling_base_GPU)

## Compile as C++14
add_compile_options(-std=c++14)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

## Find CUDA
find_package(CUDA 11 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PYTHON_INCLUDE_DIRS}
                    )

cuda_add_executable(
    ${PROJECT_NAME} main.cpp
    src/MCMPC_CartPole.cu
)

target_link_libraries(${PROJECT_NAME}
    ${PYTHON_LIBRARIES}
    ${CUDA_LIBRARIES}
    Eigen3::Eigen
)