cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER mpicc CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER mpicxx CACHE STRING "C++ compiler" FORCE)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

project(parallel_algorithms CXX)

set(CMAKE_CXX_FLAGS "-DNUMBER=2000 -DPRECISION=6 -DMODE=1 -DMAX_TEMP=1e-1")

include_directories(/opt/homebrew/include)
add_executable(hw-2 hw-2.cpp)
