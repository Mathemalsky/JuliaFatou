cmake_minimum_required(VERSION 3.10)

project(JuliaFatou LANGUAGES CXX CUDA)

set(CMAKE_CXX_COMPILER g++)

set(CXXFLAGS= "-c++17 -g -pedantic -Wall -Wextra -march=native -O3")

# specify where the output should be compiled
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/build/)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_HOME_DIRECTORY}/bin)

# lists all sourcefiles to be compiled with the project
file(GLOB SOURCES "src/*.cpp")

#lists all header files to be included in the project
file(GLOB HEADERS "include/*.hpp")

add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

target_include_directories(${PROJECT_NAME} PUBLIC include ${CMAKE_LIBRARY_OUTPUT_DIRECTORY})

# add libary with cuda code
add_subdirectory(lib/Julia)

# find PNG writer
find_package(PNGwriter 0.7.0)

# find Qt
find_package(Qt5 COMPONENTS Widgets REQUIRED)

# needed for QT::Widgets
set(CMAKE_AUTOUIC ON)
set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTORCC ON)

target_link_libraries(${PROJECT_NAME} PRIVATE PNGwriter::PNGwriter Julia Qt5::Widgets )

add_dependencies(${PROJECT_NAME} Julia)
