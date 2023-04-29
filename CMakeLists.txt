cmake_minimum_required(VERSION 3.0)

project(optimization)

set(CMAKE_CXX_STANDARD 14)

find_package(Ceres REQUIRED)

add_executable(line src/line_problem.cpp)
add_executable(cylinder src/cylinder_problem.cpp)


target_link_libraries(line Ceres::ceres)
target_link_libraries(cylinder Ceres::ceres)
