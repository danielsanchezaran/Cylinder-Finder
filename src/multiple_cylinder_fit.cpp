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
  ThreadPool thread_pool(10);

  // Define the vector to hold the PointCloud pointers
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cylinder_clouds;
  std::vector<std::future<Eigen::VectorXd>> future_cylinder_params;
  // Create new PointCloud instances and add their pointers to the vector
  int n_point_clouds = 20;
  for (int i = 0; i < n_point_clouds; ++i) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    cylinder_clouds.push_back(cloud);
  }

  std::random_device rd;
  std::mt19937 generator(rd());

  std::uniform_real_distribution<double> distribution_radius(0.05, 15.);
  std::uniform_real_distribution<double> distribution_ratio(1.1, 1.8);
  std::uniform_real_distribution<double> distribution_center(-30., 30.);
  std::uniform_int_distribution<int> distribution_points(100, 1000);

  {
    TimeIT t("Threads");
    for (int i = 0; i < n_point_clouds; ++i) {
      double axis1 = distribution_center(generator);
      double axis2 = distribution_center(generator);
      double axis3 = distribution_center(generator);

      if (axis1 == 0.0 && axis2 == 0.0 && axis3 == 0.0) axis1 = 1.0;
      Eigen::Vector3d axis(axis1, axis2, axis3);
      axis.normalize();

      double center_x = distribution_center(generator);
      double center_y = distribution_center(generator);
      double center_z = distribution_center(generator);
      Eigen::Vector3d center(center_x, center_y, center_z);


      double radius = distribution_radius(generator);
      double height = distribution_ratio(generator) * radius;
      int n_points = distribution_points(generator);

      thread_pool.enqueue([=] {
        generate_cylinder_points<pcl::PointXYZ>(n_points, axis, center, radius,
                                                height, cylinder_clouds[i]);
      });
    }

    thread_pool.waitUntilDone();

    for (int i = 0; i < n_point_clouds; ++i) {
      // auto future_param =
      future_cylinder_params.emplace_back(thread_pool.enqueue_result([=]() {
        return find_cylinder_projection_ransac<pcl::PointXYZ>(
            cylinder_clouds[i]);
        // return find_cylinder_model<pcl::PointXYZ>(cylinder_clouds[i]);
      }));
    }
    thread_pool.waitUntilDone();
  }
  // Create the PCLVisualizer
  pcl::visualization::PCLVisualizer viewer("3D Viewer");
  viewer.setWindowName("3D Viewer");  // Set a window name for the viewer

  for (int i = 0; i < n_point_clouds; ++i) {
    auto x = future_cylinder_params[i].get();
    adjust_cylinder_model_to_points<pcl::PointXYZ>(cylinder_clouds[i], x);

    // Extract the cylinder parameters from the x vector
    Eigen::Vector3d axis_ = x.segment<3>(0);
    Eigen::Vector3d center_ = x.segment<3>(3);
    double radius_ = x(6);

    pcl::ModelCoefficients::Ptr cylinder_coefficients(
        new pcl::ModelCoefficients);

    cylinder_coefficients->values.resize(7);
    cylinder_coefficients->values[0] = static_cast<float>(center_.x());
    cylinder_coefficients->values[1] = static_cast<float>(center_.y());
    cylinder_coefficients->values[2] = static_cast<float>(center_.z());
    cylinder_coefficients->values[3] = static_cast<float>(axis_.x());
    cylinder_coefficients->values[4] = static_cast<float>(axis_.y());
    cylinder_coefficients->values[5] = static_cast<float>(axis_.z());
    cylinder_coefficients->values[6] = static_cast<float>(radius_);

    viewer.addCylinder(*cylinder_coefficients, "cylinder" + std::to_string(i));

    // Set the color of the cylinder (e.g., red color)
    double red = (1.0 / n_point_clouds) * (n_point_clouds - i);
    double green = (1.0 - (static_cast<double>(i + 1) / n_point_clouds)) * 0.5;
    double blue = 0.5;

    viewer.setShapeRenderingProperties(pcl::visualization::PCL_VISUALIZER_COLOR,
                                       red, green, blue,
                                       "cylinder" + std::to_string(i));

    // Set the transparency of the cylinder (0.0 is completely transparent, 1.0
    // is opaque)
    double transparency = 0.5;  // You can adjust this value as needed
    viewer.setShapeRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_OPACITY, transparency,
        "cylinder" + std::to_string(i));
  }


  for (int i = 0; i < n_point_clouds; ++i) {
    std::string name("cloud" + std::to_string(i));
    viewer.addPointCloud<pcl::PointXYZ>(cylinder_clouds[i], name);
  }
  draw_origin(viewer);
  viewer.spin();
  return 0;
}
