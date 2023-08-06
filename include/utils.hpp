#ifndef _UTILS_HPP_
#define _UTILS_HPP_

#include <pcl/visualization/pcl_visualizer.h>

#include <chrono>
#include <iostream>
#include <string>

class TimeIT {
 private:
  std::chrono::time_point<std::chrono::steady_clock> t_start;
  std::string instance;

 public:
  explicit TimeIT(std::string instance) : instance(instance) {
    t_start = std::chrono::steady_clock::now();
  }

  ~TimeIT() {
    std::chrono::time_point<std::chrono::steady_clock> t_end =
        std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = t_end - t_start;
    std::cout << "Elapsed time for " << instance << ": " << duration.count()
              << " (s)\n";
  }
};

static void draw_origin(pcl::visualization::PCLVisualizer& viewer) {
  // Create the coordinate frame manually
  pcl::PointXYZ origin(0, 0, 0);
  pcl::PointXYZ x_axis(1, 0, 0);
  pcl::PointXYZ y_axis(0, 1, 0);
  pcl::PointXYZ z_axis(0, 0, 1);

  viewer.addLine(origin, x_axis, 1.0, 0.0, 0.0, "x_axis");
  viewer.addLine(origin, y_axis, 0.0, 1.0, 0.0, "y_axis");
  viewer.addLine(origin, z_axis, 0.0, 0.0, 1.0, "z_axis");
};

#endif  // _UTILS_HPP_