cmake_minimum_required(VERSION 4.0)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

project(cuda_playground CXX CUDA)

set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories("src")
#include_directories("src-dcore")
include_directories("dataset")

file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp")
#file(GLOB_RECURSE SOURCES_DCORE RELATIVE ${CMAKE_SOURCE_DIR} "src-dcore/*.cpp")

add_executable(cuda_playground ${SOURCES} main.cu)
#add_executable(cuda_playground ${SOURCES_DCORE} main_dcore.cpp)

set_target_properties(cuda_playground PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
