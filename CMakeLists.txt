cmake_minimum_required(VERSION 2.8.12)
project(dmaps)

include_directories("${PROJECT_SOURCE_DIR}/include")
include_directories("${PROJECT_SOURCE_DIR}/include/pybind11/include")
SET(SOURCES 
	dmaps/distance_matrix.cpp
)

add_subdirectory(include/pybind11)
pybind11_add_module(dmaps ${SOURCES})