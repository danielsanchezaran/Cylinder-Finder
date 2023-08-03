#include <pcl/visualization/pcl_visualizer.h>

#include <chrono>
#include <iostream>
#include <limits>
#include <string>

#include "cylinder_fitting.hpp"
#include "ransac.hpp"

class TimeIT {
 private:
  std::chrono::time_point<std::chrono::steady_clock> t_start;
  std::string instance;
 public:
  explicit TimeIT(std::string instance) : instance(instance){ t_start = std::chrono::steady_clock::now(); }

  ~TimeIT() {
    std::chrono::time_point<std::chrono::steady_clock> t_end =
        std::chrono::steady_clock::now();
    std::chrono::duration<double> duration = t_end - t_start;
    std::cout << "Elapsed time for " << instance <<": "<< duration.count() << " (s)\n";
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
  pcl::PointCloud<pcl::PointXYZ>::Ptr cylinder_cloud(
      new pcl::PointCloud<pcl::PointXYZ>);
  Eigen::Vector3d axis(1, 1, 1);
  axis.normalize();
  Eigen::Vector3d center(-3, 7, -2.5);
  double radius = 3;
  double height = 2;
  int n_points = 1000;
  generate_cylinder_points<pcl::PointXYZ>(n_points, axis, center, radius,
                                          height, cylinder_cloud);

  {
    TimeIT t("Find Cylinder Ransac");
    auto x = find_cylinder_projection_ransac<pcl::PointXYZ>(cylinder_cloud);
  }
  {
    TimeIT t("Find Cylinder Ransac + Optimization");
    auto x = find_cylinder_projection_ransac<pcl::PointXYZ>(cylinder_cloud,true);
  }
  {
    TimeIT t(("Find Cylinder LSQ"));
    auto x = find_cylinder_model<pcl::PointXYZ>(cylinder_cloud);
  }
  auto x = find_cylinder_model<pcl::PointXYZ>(cylinder_cloud);
  std::cout << "Output " << x.transpose() << "\n";
  // TODO: Filter inliers before modifying
  cylinder_cloud =
      filter_cylinder_inliers<pcl::PointXYZ>(cylinder_cloud, x, 0.05);
  adjust_cylinder_model_to_points<pcl::PointXYZ>(cylinder_cloud, x);

  auto toEigen = pcl_to_eigen<pcl::PointXYZ>(cylinder_cloud);

  std::cout << "Output after adjust" << x.transpose() << "\n";

  // Create the PCLVisualizer
  pcl::visualization::PCLVisualizer viewer("3D Viewer");
  viewer.setWindowName("3D Viewer");  // Set a window name for the viewer


  // Extract the cylinder parameters from the x vector
  Eigen::Vector3d axis_ = x.segment<3>(0);
  Eigen::Vector3d center_ = x.segment<3>(3);
  double radius_ = x(6);

  auto collapsedPC = project_points_perpendicular_to_axis<pcl::PointXYZ>(
      cylinder_cloud, axis_.cast<float>());
  // Visualize the point cloud
  viewer.addPointCloud<pcl::PointXYZ>(cylinder_cloud, "cloud");
  // viewer.addPointCloud<pcl::PointXYZ>(collapsedPC, "cloud");


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

  draw_origin(viewer);

  viewer.spin();

  return 0;
}
