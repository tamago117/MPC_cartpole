#
# Copyright 2019 Gianluca Frison, Dimitris Kouzoupis, Robin Verschueren,
# Andrea Zanelli, Niels van Duijkeren, Jonathan Frey, Tommaso Sartor,
# Branimir Novoselnik, Rien Quirynen, Rezart Qelibari, Dang Doan,
# Jonas Koenemann, Yutao Chen, Tobias Schöls, Jonas Schlagenhauf, Moritz Diehl
#
# This file is part of acados.
#
# The 2-Clause BSD License
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
# this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.;
#



cmake_minimum_required(VERSION 3.10)

project(cart_pole)

# build options.
option(BUILD_ACADOS_SOLVER_LIB "Should the solver library acados_solver_cart_pole be build?" OFF)
option(BUILD_ACADOS_OCP_SOLVER_LIB "Should the OCP solver library acados_ocp_solver_cart_pole be build?" OFF)
option(BUILD_EXAMPLE "Should the example main_cart_pole be build?" OFF)
option(BUILD_SIM_EXAMPLE "Should the simulation example main_sim_cart_pole be build?" OFF)
option(BUILD_ACADOS_SIM_SOLVER_LIB "Should the simulation solver library acados_sim_solver_cart_pole be build?" OFF)

# object target names
set(MODEL_OBJ model_cart_pole)
set(OCP_OBJ ocp_cart_pole)
set(SIM_OBJ sim_cart_pole)

# model
set(MODEL_SRC
    cart_pole_model/cart_pole_expl_ode_fun.c
    cart_pole_model/cart_pole_expl_vde_forw.c
)
add_library(${MODEL_OBJ} OBJECT ${MODEL_SRC} )

# optimal control problem - mostly CasADi exports
if(${BUILD_ACADOS_SOLVER_LIB} OR ${BUILD_ACADOS_OCP_SOLVER_LIB} OR ${BUILD_EXAMPLE})
    set(OCP_SRC
        acados_solver_cart_pole.c)
    add_library(${OCP_OBJ} OBJECT ${OCP_SRC})
endif()
# for sim solver
if(${BUILD_ACADOS_SOLVER_LIB} OR ${BUILD_EXAMPLE}
    OR ${BUILD_SIM_EXAMPLE} OR ${BUILD_ACADOS_SIM_SOLVER_LIB})
    set(SIM_SRC acados_sim_solver_cart_pole.c)
    add_library(${SIM_OBJ} OBJECT ${SIM_SRC})
endif()

# for target example
set(EX_SRC main_cart_pole.c)
set(EX_EXE main_cart_pole)
set(EXTERNAL_DIR)
set(EXTERNAL_LIB)

# set some search paths for preprocessor and linker
set(ACADOS_INCLUDE_PATH /home/acados/include CACHE PATH "Define the path which contains the include directory for acados.")
set(ACADOS_LIB_PATH /home/acados/lib CACHE PATH "Define the path which contains the lib directory for acados.")

# c-compiler flags for debugging
set(CMAKE_C_FLAGS_DEBUG "-O0 -ggdb")

set(CMAKE_C_FLAGS "-fPIC -std=c99  ")
#-fno-diagnostics-show-line-numbers -g

include_directories(
   ${ACADOS_INCLUDE_PATH}
   ${ACADOS_INCLUDE_PATH}/acados
   ${ACADOS_INCLUDE_PATH}/blasfeo/include
   ${ACADOS_INCLUDE_PATH}/hpipm/include
)

# linker flags
link_directories(${ACADOS_LIB_PATH})

# link to libraries
if(UNIX)
    link_libraries(acados hpipm blasfeo m )
else()
    link_libraries(acados hpipm blasfeo )
endif()

# the targets

# bundled_shared_lib
if(${BUILD_ACADOS_SOLVER_LIB})
    set(LIB_ACADOS_SOLVER acados_solver_cart_pole)
    add_library(${LIB_ACADOS_SOLVER} SHARED $<TARGET_OBJECTS:${MODEL_OBJ}> $<TARGET_OBJECTS:${OCP_OBJ}>
        $<TARGET_OBJECTS:${SIM_OBJ}>)
    install(TARGETS ${LIB_ACADOS_SOLVER} DESTINATION ${CMAKE_INSTALL_PREFIX})
endif(${BUILD_ACADOS_SOLVER_LIB})

# ocp_shared_lib
if(${BUILD_ACADOS_OCP_SOLVER_LIB})
    set(LIB_ACADOS_OCP_SOLVER acados_ocp_solver_cart_pole)
    add_library(${LIB_ACADOS_OCP_SOLVER} SHARED $<TARGET_OBJECTS:${MODEL_OBJ}> $<TARGET_OBJECTS:${OCP_OBJ}>)
    # Specify libraries or flags to use when linking a given target and/or its dependents.
    target_link_libraries(${LIB_ACADOS_OCP_SOLVER} PRIVATE ${EXTERNAL_LIB})
    target_link_directories(${LIB_ACADOS_OCP_SOLVER} PRIVATE ${EXTERNAL_DIR})
    install(TARGETS ${LIB_ACADOS_OCP_SOLVER} DESTINATION ${CMAKE_INSTALL_PREFIX})
endif(${BUILD_ACADOS_OCP_SOLVER_LIB})

# example
if(${BUILD_EXAMPLE})
    add_executable(${EX_EXE} ${EX_SRC} $<TARGET_OBJECTS:${MODEL_OBJ}> $<TARGET_OBJECTS:${OCP_OBJ}>
        $<TARGET_OBJECTS:${SIM_OBJ}>)
    install(TARGETS ${EX_EXE} DESTINATION ${CMAKE_INSTALL_PREFIX})
endif(${BUILD_EXAMPLE})

# example_sim
if(${BUILD_SIM_EXAMPLE})
    set(EX_SIM_SRC main_sim_cart_pole.c)
    set(EX_SIM_EXE main_sim_cart_pole)
    add_executable(${EX_SIM_EXE} ${EX_SIM_SRC} $<TARGET_OBJECTS:${MODEL_OBJ}> $<TARGET_OBJECTS:${SIM_OBJ}>)
    install(TARGETS ${EX_SIM_EXE} DESTINATION ${CMAKE_INSTALL_PREFIX})
endif(${BUILD_SIM_EXAMPLE})

# sim_shared_lib
if(${BUILD_ACADOS_SIM_SOLVER_LIB})
    set(LIB_ACADOS_SIM_SOLVER acados_sim_solver_cart_pole)
    add_library(${LIB_ACADOS_SIM_SOLVER} SHARED $<TARGET_OBJECTS:${MODEL_OBJ}> $<TARGET_OBJECTS:${SIM_OBJ}>)
    install(TARGETS ${LIB_ACADOS_SIM_SOLVER} DESTINATION ${CMAKE_INSTALL_PREFIX})
endif(${BUILD_ACADOS_SIM_SOLVER_LIB})

