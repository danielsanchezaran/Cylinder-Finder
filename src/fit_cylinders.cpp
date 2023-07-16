#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>

#include "cylinder_fitting.hpp"

int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Vector3d axis(0, 0.707, 0.707);
  Eigen::Vector3d center(0, 0, 0);
  double radius = 2;
  double height = 2;
  int n_points = 1000;
  generate_cylinder_points<pcl::PointXYZ>(n_points, axis, center, radius,
                                          height, cylinder_cloud);

  auto x = find_cylinder<pcl::PointXYZ>(cylinder_cloud);

  std::cout << "Output " << x.transpose() << "\n";

  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("3D Viewer"));
  viewer->addPointCloud<pcl::PointXYZ>(cylinder_cloud, "cloud");

  // Extract the cylinder parameters from the x vector
  Eigen::Vector3d axis_ = x.segment<3>(0);
  Eigen::Vector3d center_ = x.segment<3>(3);
  double radius_ = x(6);

  pcl::ModelCoefficients::Ptr cylinder_coefficients(new pcl::ModelCoefficients);
  cylinder_coefficients->values.resize(7);
  cylinder_coefficients->values[0] = static_cast<float>(center_.x());
  cylinder_coefficients->values[1] = static_cast<float>(center_.y());
  cylinder_coefficients->values[2] = static_cast<float>(center_.z());
  cylinder_coefficients->values[3] = static_cast<float>(axis_.x());
  cylinder_coefficients->values[4] = static_cast<float>(axis_.y());
  cylinder_coefficients->values[5] = static_cast<float>(axis_.z());
  cylinder_coefficients->values[6] = static_cast<float>(radius_);

  viewer->addCylinder(*cylinder_coefficients, "cylinder");

  // Visualize the results
  viewer->spin();

  std::cout << x.transpose() << std::endl;
  return 0;
}
