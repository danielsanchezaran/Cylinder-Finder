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

  // Create the PCLVisualizer
  pcl::visualization::PCLVisualizer viewer("3D Viewer");
  viewer.setWindowName("3D Viewer");  // Set a window name for the viewer

  // Visualize the point cloud
  viewer.addPointCloud<pcl::PointXYZ>(cylinder_cloud, "cloud");

  // Extract the cylinder parameters from the x vector
  Eigen::Vector3d axis_ = x.segment<3>(0);
  Eigen::Vector3d center_ = x.segment<3>(3);
  double radius_ = x(6);

  pcl::ModelCoefficients::Ptr cylinder_coefficients(new pcl::ModelCoefficients);
  // Compute the top point of the cylinder based on the desired height
  Eigen::Vector3d top = center_ + axis_ * height * -2.0;

  cylinder_coefficients->values.resize(7);
  cylinder_coefficients->values[0] = static_cast<float>(center_.x());
  cylinder_coefficients->values[1] = static_cast<float>(center_.y());
  cylinder_coefficients->values[2] = static_cast<float>(center_.z());
  cylinder_coefficients->values[3] = static_cast<float>(axis_.x());
  cylinder_coefficients->values[4] = static_cast<float>(axis_.y());
  cylinder_coefficients->values[5] = static_cast<float>(axis_.z());
  cylinder_coefficients->values[6] = static_cast<float>(radius_);

  // Update the top point in the cylinder coefficients
  cylinder_coefficients->values[0] = static_cast<float>(top.x());
  cylinder_coefficients->values[1] = static_cast<float>(top.y());
  cylinder_coefficients->values[2] = static_cast<float>(top.z());

  viewer.addCylinder(*cylinder_coefficients, "cylinder");

  // Set the color of the cylinder (e.g., red color)
  double red = 1.0;
  double green = 0.0;
  double blue = 0.0;
  viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                     red, green, blue, "cylinder");

  // Set the transparency of the cylinder (0.0 is completely transparent, 1.0 is
  // opaque)
  double transparency = 0.5;  // You can adjust this value as needed
  viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                     transparency, "cylinder");

  // Create the coordinate frame manually
  pcl::PointXYZ origin(0, 0, 0);
  pcl::PointXYZ x_axis(1, 0, 0);
  pcl::PointXYZ y_axis(0, 1, 0);
  pcl::PointXYZ z_axis(0, 0, 1);

  // viewer.addLine(origin, x_axis, 1.0, 0.0, 0.0, "x_axis");
  // viewer.addLine(origin, y_axis, 0.0, 1.0, 0.0, "y_axis");
  // viewer.addLine(origin, z_axis, 0.0, 0.0, 1.0, "z_axis");

  viewer.spin();

  std::cout << x.transpose() << std::endl;
  return 0;
}
