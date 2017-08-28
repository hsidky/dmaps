cmake_minimum_required(VERSION 2.8.12)
project(dmaps)

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/include/pybind11/include")
set(SOURCES 
	dmaps/diffusion_map.cpp
	dmaps/distance_matrix.cpp
	dmaps/metrics.cpp
)

find_package(OpenMP QUIET)
if(OPENMP_FOUND)
	set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")

add_subdirectory(include/pybind11)
pybind11_add_module(dmaps ${SOURCES} dmaps/bindings.cpp)