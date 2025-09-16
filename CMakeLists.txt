cmake_minimum_required(VERSION 4.0)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

project(d_core_cuda CXX CUDA)

#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")

include_directories("src")
#include_directories("src-dcore")
include_directories("dataset")

file(GLOB_RECURSE SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp")
#file(GLOB_RECURSE SOURCES_DCORE RELATIVE ${CMAKE_SOURCE_DIR} "src-dcore/*.cpp")

add_executable(d_core_cuda ${SOURCES} main.cu)
#add_executable(d_core_cuda ${SOURCES_DCORE} main_dcore.cpp)

set_target_properties(d_core_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
