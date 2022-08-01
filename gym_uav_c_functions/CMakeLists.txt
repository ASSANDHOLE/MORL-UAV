cmake_minimum_required(VERSION 3.10)
project(c_functions)

set(CMAKE_CXX_STANDARD 14)

# add_executable(${PROJECT_NAME} main.cpp)
add_library(${PROJECT_NAME} SHARED main.cpp)
