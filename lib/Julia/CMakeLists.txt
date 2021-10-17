set(CMAKE_CUDA_COMPILER nvcc)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --compiler-options -O2 -g")

file(GLOB SOURCES
    "src/*.cpp"
    "src/*.cu")
file(GLOB HEADERS
    "include/*.hpp")

add_library(Julia STATIC ${SOURCES} ${HEADERS})
target_include_directories(Julia PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
