cmake_minimum_required(VERSION 3.30)
project(cppThermalCamera VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

add_executable(cppThermalCamera main.cpp)

configure_file(
        "${PROJECT_SOURCE_DIR}/version.h.in"
        "${PROJECT_BINARY_DIR}/version.h"
)

target_include_directories(cppThermalCamera PUBLIC "${PROJECT_BINARY_DIR}")
include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cppThermalCamera ${OpenCV_LIBS})
