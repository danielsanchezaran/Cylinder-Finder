#include <pcl/visualization/pcl_visualizer.h>

#include <chrono>
#include <future>
#include <iostream>
#include <limits>
#include <string>

#include "cylinder_fitting.hpp"
#include "ransac.hpp"
#include "thread_pool.hpp"

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

void draw_origin(pcl::visualization::PCLVisualizer& viewer) {
  // Create the coordinate frame manually
  pcl::PointXYZ origin(0, 0, 0);
  pcl::PointXYZ x_axis(1, 0, 0);
  pcl::PointXYZ y_axis(0, 1, 0);
  pcl::PointXYZ z_axis(0, 0, 1);

  viewer.addLine(origin, x_axis, 1.0, 0.0, 0.0, "x_axis");
  viewer.addLine(origin, y_axis, 0.0, 1.0, 0.0, "y_axis");
  viewer.addLine(origin, z_axis, 0.0, 0.0, 1.0, "z_axis");
}


int main() {
  ThreadPool thread_pool(4);

  // Define the vector to hold the PointCloud pointers
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cylinder_clouds;

  // Create new PointCloud instances and add their pointers to the vector
  int n_point_clouds = 8;
  for (int i = 0; i < n_point_clouds; ++i) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    cylinder_clouds.push_back(cloud);
  }

  std::random_device rd;
  std::mt19937 generator(rd());

  for (int i = 0; i < n_point_clouds; ++i) {
    std::uniform_int_distribution<int> distribution(1, 10);
    int axis1 = distribution(generator);
    int axis2 = distribution(generator);
    int axis3 = distribution(generator);

    Eigen::Vector3d axis(axis1, axis2, axis3);
    axis.normalize();

    std::uniform_real_distribution<double> distribution_center(-10., 10.);
    double center_x = distribution_center(generator);
    double center_y = distribution_center(generator);
    double center_z = distribution_center(generator);

    Eigen::Vector3d center(center_x, center_y, center_z);
    double radius = 3;
    double height = 2;
    int n_points = 1000;

    thread_pool.enqueue([=] {
      generate_cylinder_points_copy<pcl::PointXYZ>(n_points, axis, center, radius,
                                              height, cylinder_clouds[i]);
    });
  }

  thread_pool.waitUntilDone();

  // Create the PCLVisualizer
  pcl::visualization::PCLVisualizer viewer("3D Viewer");
  viewer.setWindowName("3D Viewer");  // Set a window name for the viewer
  for (int i = 0; i < n_point_clouds; ++i) {
    std::string name("cloud" + std::to_string(i));
    viewer.addPointCloud<pcl::PointXYZ>(cylinder_clouds[i], name);
  }
  draw_origin(viewer);
  viewer.spin();
  return 0;
}
