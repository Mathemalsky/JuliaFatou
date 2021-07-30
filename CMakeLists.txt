cmake_minimum_required(VERSION 3.10)

project(JuliaFatou LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER nvcc)

set(CXXFLAGS= "-c++17 -g -pedantic -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -O2 -g")

#find_package(CUDAToolkit 11.3 REQUIRED)

# specify where the output should be compiled
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

set(SOURCES
        main.cpp
        printimage.cpp
        calculate.cu
    )
set(HEADERS
        calculate.hpp
        printimage.hpp
    )

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
