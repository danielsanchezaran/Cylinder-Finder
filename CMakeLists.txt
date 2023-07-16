cmake_minimum_required(VERSION 3.0)

project(optimization)

set(CMAKE_CXX_STANDARD 14)

# Find Ceres and PCL packages
find_package(Ceres REQUIRED)
find_package(PCL 1.7 REQUIRED)

# Add include directories for Ceres and PCL
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(./include ./src)

# Add executable targets
add_executable(line_app src/line_problem.cpp)
add_executable(cylinder_app src/fit_cylinders.cpp)

# Link the targets with Ceres and PCL libraries
target_link_libraries(line_app Ceres::ceres)
target_link_libraries(cylinder_app Ceres::ceres ${PCL_LIBRARIES})
