cmake_minimum_required(VERSION 3.13)
project(heat C)

set(CMAKE_C_STANDARD 99)

add_executable(heat main.c)

find_package(MPI REQUIRED)

set(CMAKE_CXX_COMPILE_FLAGS ${CMAKE_CXX_COMPILE_FLAGS} ${MPI_COMPILE_FLAGS})
set(CMAKE_CXX_LINK_FLAGS ${CMAKE_CXX_LINK_FLAGS} ${MPI_LINK_FLAGS})

include_directories(MPI_INCLUDE_PATH)
target_link_libraries(heat -lm ${MPI_LIBRARIES})