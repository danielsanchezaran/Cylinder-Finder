cmake_minimum_required(VERSION 3.0)

project(optimization)

set(CMAKE_CXX_STANDARD 17)

# Find Ceres and PCL packages
find_package(Ceres REQUIRED)
find_package(PCL 1.7 REQUIRED)

# Add include directories for Ceres, PCL, and ThreadPool
include_directories(${CERES_INCLUDE_DIRS})
include_directories(${PCL_INCLUDE_DIRS})
include_directories(./include ./src ./ThreadPool/include)

# Add executable targets
add_executable(line_app src/line_problem.cpp)
add_executable(cylinder_app src/fit_cylinders.cpp ./ThreadPool/src/thread_pool.cpp ./ThreadPool/src/thread_safe_counter.cpp)
add_executable(multiple_cylinder_fit src/multiple_cylinder_fit.cpp ./ThreadPool/src/thread_pool.cpp ./ThreadPool/src/thread_safe_counter.cpp)
add_executable(clustering_with_cylinders src/clustering_with_cylinders.cpp ./ThreadPool/src/thread_pool.cpp ./ThreadPool/src/thread_safe_counter.cpp)

# Add the ransac_test target
add_executable(ransac_test tests/ransac_test.cpp)
add_executable(clustering_test tests/clustering_test.cpp)


# Link the targets with Ceres and PCL libraries
target_link_libraries(line_app ${CERES_LIBRARIES})
target_link_libraries(cylinder_app ${CERES_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(multiple_cylinder_fit ${CERES_LIBRARIES} ${PCL_LIBRARIES})
target_link_libraries(clustering_with_cylinders ${CERES_LIBRARIES} ${PCL_LIBRARIES})

# Link the ransac_test with Google Test library
target_link_libraries(ransac_test ${CERES_LIBRARIES} ${PCL_LIBRARIES} gtest gtest_main)
target_link_libraries(clustering_test ${CERES_LIBRARIES} ${PCL_LIBRARIES})


# Add the test to CTest
enable_testing()
add_test(NAME RANSACTest COMMAND ransac_test)
