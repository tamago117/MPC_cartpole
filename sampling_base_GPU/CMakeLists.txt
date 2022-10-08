cmake_minimum_required(VERSION 3.4...3.18)
project(sampling_base_GPU)

## Compile as C++11
#add_compile_options(-std=c++11)
## ignore warnings

set(CMAKE_CXX_FLAGS                " ${CMAKE_CXX_FLAGS_INIT} -w -std=c++11 -fPIC")
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_POSITION_INDEPENDENT_CODE ON)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -dc")

## Find CUDA
find_package(CUDA 11 REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PYTHON_INCLUDE_DIRS}
                    ./pybind11/include
                    )



cuda_add_executable(${PROJECT_NAME}
    main.cpp
    src/CartPole.cu
    src/cuda_utils.cu
    src/MCMPC_CartPole.cu
)
target_link_libraries(${PROJECT_NAME}
    ${PYTHON_LIBRARIES}
    ${CUDA_LIBRARIES}
)



add_subdirectory(pybind11)
cuda_add_library(mcmpc_CartPole SHARED
    src/MCMPC_CartPole_pybind.cpp
    src/CartPole.cu
    src/cuda_utils.cu
    src/MCMPC_CartPole.cu
)
target_link_libraries(mcmpc_CartPole
    ${PYTHON_LIBRARIES}
    ${CUDA_LIBRARIES}
)

set_target_properties(mcmpc_CartPole PROPERTIES PREFIX "")