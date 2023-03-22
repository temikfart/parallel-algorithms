cmake_minimum_required(VERSION 3.16)

set(CMAKE_C_COMPILER mpicc CACHE STRING "C compiler" FORCE)
set(CMAKE_CXX_COMPILER mpicxx CACHE STRING "C++ compiler" FORCE)
set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)

project(parallel_algorithms CXX)

include_directories(/opt/homebrew/include)

# 1
add_subdirectory(sem-1)
add_subdirectory(hw-1)
