#include <pcl/visualization/pcl_visualizer.h>

#include <iostream>
#include <limits>

#include "cylinder_fitting.hpp"

double projection_of_point_in_axis(const Eigen::Vector3d& direction,
                                   const Eigen::Vector3d& line_point,
                                   const Eigen::Vector3d& point) {
  auto line_point_to_point = point - line_point;
  return line_point_to_point.dot(direction);
}

template <typename PointT>
void adjust_cylinder_model_to_points(
    typename pcl::PointCloud<PointT>::Ptr& cylinder_cloud,
    Eigen::VectorXd& cylinder_coefficients) {
  // Extract the cylinder parameters from the x vector
  Eigen::Vector3d axis_ = cylinder_coefficients.segment<3>(0);
  Eigen::Vector3d center_ = cylinder_coefficients.segment<3>(3);

  axis_.normalize();
  double max_projection = std::numeric_limits<double>::lowest();
  double min_projection = std::numeric_limits<double>::max();
  int c = 0;
  for (auto point : cylinder_cloud->points) {
    Eigen::Vector3d point_{point.x, point.y, point.z};
    double projection = projection_of_point_in_axis(axis_, center_, point_);
    max_projection =
        (projection > max_projection) ? projection : max_projection;
    min_projection =
        (projection < min_projection) ? projection : min_projection;
  }
  Eigen::Vector3d new_center_ = center_ + axis_ * min_projection;
  Eigen::Vector3d new_axis_ = axis_ * (max_projection - min_projection);
  cylinder_coefficients.segment<3>(0) = new_axis_;
  cylinder_coefficients.segment<3>(3) = new_center_;
}

template <typename PointT>
typename pcl::PointCloud<PointT>::Ptr filter_cylinder_inliers(
    typename pcl::PointCloud<PointT>::Ptr& cylinder_cloud,
    Eigen::VectorXd& cylinder_coefficients, double radius_percentage = 0.05) {
  // Extract the cylinder parameters from the x vector
  Eigen::Vector3d axis_ = cylinder_coefficients.segment<3>(0);
  axis_.normalize();
  Eigen::Vector3d center_ = cylinder_coefficients.segment<3>(3);
  double radius = cylinder_coefficients(6);

  typename pcl::PointCloud<PointT>::Ptr filtered_cylinder_cloud(
      new pcl::PointCloud<PointT>);

  filtered_cylinder_cloud->points.resize(cylinder_cloud->points.size());
  int point_count = 0;
  for (auto point : cylinder_cloud->points) {
    Eigen::Vector3d point_{point.x, point.y, point.z};
    double projection = projection_of_point_in_axis(axis_, center_, point_);
    double distance_to_axis = (point_ - (center_ + projection * axis_)).norm();

    if (std::abs(distance_to_axis - radius) < radius * radius_percentage)
      filtered_cylinder_cloud->points[point_count++] = point;
  }
  filtered_cylinder_cloud->resize(point_count);
  return filtered_cylinder_cloud;
}

int main() {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Vector3d axis(1, 0, 0);
  axis.normalize();
  Eigen::Vector3d center(0, 0, 0);
  double radius = 2;
  double height = 2;
  int n_points = 1000;
  generate_cylinder_points<pcl::PointXYZ>(n_points, axis, center, radius,
                                          height, cylinder_cloud);

  auto x = find_cylinder<pcl::PointXYZ>(cylinder_cloud);
  std::cout << "Output " << x.transpose() << "\n";
  // TODO: Filter inliers before modifying
  cylinder_cloud =
      filter_cylinder_inliers<pcl::PointXYZ>(cylinder_cloud, x, 0.05);
  adjust_cylinder_model_to_points<pcl::PointXYZ>(cylinder_cloud, x);

  std::cout << "Output after adjust" << x.transpose() << "\n";

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

  cylinder_coefficients->values.resize(7);
  cylinder_coefficients->values[0] = static_cast<float>(center_.x());
  cylinder_coefficients->values[1] = static_cast<float>(center_.y());
  cylinder_coefficients->values[2] = static_cast<float>(center_.z());
  cylinder_coefficients->values[3] = static_cast<float>(axis_.x());
  cylinder_coefficients->values[4] = static_cast<float>(axis_.y());
  cylinder_coefficients->values[5] = static_cast<float>(axis_.z());
  cylinder_coefficients->values[6] = static_cast<float>(radius_);

  viewer.addCylinder(*cylinder_coefficients, "cylinder");

  // Set the color of the cylinder (e.g., red color)
  double red = 0.3;
  double green = 0.0;
  double blue = 0.3;
  viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                     red, green, blue, "cylinder");

  // Set the transparency of the cylinder (0.0 is completely transparent, 1.0 is
  // opaque)
  double transparency = 0.2;  // You can adjust this value as needed
  viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_OPACITY,
                                     transparency, "cylinder");

  // Create the coordinate frame manually
  pcl::PointXYZ origin(0, 0, 0);
  pcl::PointXYZ x_axis(1, 0, 0);
  pcl::PointXYZ y_axis(0, 1, 0);
  pcl::PointXYZ z_axis(0, 0, 1);

  viewer.addLine(origin, x_axis, 1.0, 0.0, 0.0, "x_axis");
  viewer.addLine(origin, y_axis, 0.0, 1.0, 0.0, "y_axis");
  viewer.addLine(origin, z_axis, 0.0, 0.0, 1.0, "z_axis");

  viewer.spin();

  std::cout << x.transpose() << std::endl;
  return 0;
}
