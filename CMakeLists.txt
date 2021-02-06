cmake_minimum_required(VERSION 3.10)

project(JuliaFatou)

# specify where the output should be compiled
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

set(CXX_FLAGS= "-c++17 -O3 -g")

set(SOURCES
        main.cpp
        printimage.cpp
    )
set(HEADERS
        printimage.hpp
    )

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})
