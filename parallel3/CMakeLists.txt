cmake_minimum_required(VERSION 3.13)
project(parallel3)

set(CMAKE_CXX_STANDARD 17)

# added -fopenmp
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -fopenmp")

add_executable(parallel3 main.cpp)