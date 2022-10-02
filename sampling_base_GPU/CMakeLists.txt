cmake_minimum_required(VERSION 3.8)
project(sampling_base_GPU)

## Compile as C++14
add_compile_options(-std=c++14)

#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_BUILD_TYPE Release)
#set(GENCODE -gencode=arch=compute_10,code=compute_10)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CUDA_SEPARABLE_COMPILATION ON)
#set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -rdc=true")
#set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -dc")

## Find CUDA
find_package(CUDA 11 REQUIRED)
find_package(PythonLibs REQUIRED)

include_directories(${PROJECT_SOURCE_DIR}/include
                    ${PYTHON_INCLUDE_DIRS}
                    )



cuda_add_executable(${PROJECT_NAME}
    main.cpp
    src/CartPole.cu
    src/MCMPC_CartPole.cu
)



target_link_libraries(${PROJECT_NAME}
    ${PYTHON_LIBRARIES}
    ${CUDA_LIBRARIES}
)