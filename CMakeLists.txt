cmake_minimum_required(VERSION 3.10)

project(JuliaFatou LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER g++)
set(CMAKE_CUDA_COMPILER nvcc)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -g -pedantic -Wall -Wextra")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -O2 -g")

# specify where the output should be compiled
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)


# lists all sourcefiles to be compiled with the project
file(GLOB SOURCES "src/*.cpp" "src/*.cu")

#lists all header files to be included in the project
file(GLOB HEADERS "include/*.hpp")

add_subdirectory(lib)
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PUBLIC include)

find_package(PNGwriter 0.7.0)

if(PNGwriter_FOUND)
  target_link_libraries(${PROJECT_NAME} PRIVATE PNGwriter::PNGwriter)
endif(PNGwriter_FOUND)

find_package(glfw3 REQUIRED)

target_link_libraries (${PROJECT_NAME} PRIVATE glfw IMGUI)
