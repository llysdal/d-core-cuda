cmake_minimum_required(VERSION 3.22)

set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CXX_STANDARD 20)

project(d_core_cuda LANGUAGES CXX CUDA)

find_package(OpenMP REQUIRED)

include_directories("src")
#include_directories("src-dcore")
include_directories("dataset")

file(GLOB_RECURSE CPP_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cpp")
file(GLOB_RECURSE CUDA_SOURCES RELATIVE ${CMAKE_SOURCE_DIR} "src/*.cu")
#file(GLOB_RECURSE SOURCES_DCORE RELATIVE ${CMAKE_SOURCE_DIR} "src-dcore/*.cpp")

add_executable(d_core_cuda ${CPP_SOURCES} ${CUDA_SOURCES} main.cu)
#add_executable(d_core_cuda ${CPP_SOURCES} main.cpp)
#add_executable(d_core_cuda ${SOURCES_DCORE} main_dcore.cpp)

#set_target_properties(d_core_cuda PROPERTIES CUDA_ARCHITECTURES "89")
target_compile_options(d_core_cuda PRIVATE ${OpenMP_CXX_FLAGS})
target_link_libraries(d_core_cuda PRIVATE OpenMP::OpenMP_CXX)
set_target_properties(d_core_cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
