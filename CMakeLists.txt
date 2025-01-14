cmake_minimum_required(VERSION 3.30)
project(cppThermalCamera)

set(CMAKE_CXX_STANDARD 20)

find_package(OpenCV REQUIRED)

add_executable(cppThermalCamera main.cpp)

include_directories(${OpenCV_INCLUDE_DIRS})
target_link_libraries(cppThermalCamera ${OpenCV_LIBS})