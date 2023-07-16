#include <ceres/ceres.h>

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

  auto cylinder_centroid = computeCentroid<pcl::PointXYZ>(cylinder_cloud);
  Eigen::Vector3f centroid3f;
  centroid3f << cylinder_centroid(0), cylinder_centroid(1),
      cylinder_centroid(2);
  std::cout << "Centroid " << cylinder_centroid.transpose() << "\n";

  pcl::PCA<pcl::PointXYZ> pca;
  pca.setInputCloud(cylinder_cloud);
  std::cout << "PCA: " << pca.getEigenVectors() << "\n";
  Eigen::Vector3f main_axis = pca.getEigenVectors().col(2);

  auto projection =
      project_points_perpendicular_to_axis(*cylinder_cloud, main_axis);
  pcl::PointCloud<pcl::PointXYZ>::Ptr projection_cloud_ptr =
      projection.makeShared();
  auto radius_approx = compute_cylinder_radius<pcl::PointXYZ>(
      projection_cloud_ptr, main_axis, centroid3f);
  std::cout << "Radius " << radius_approx << "\n";
  ceres::Problem problem;
  Eigen::VectorXd x(7);

  // x << axis,center,radius;
  x << main_axis.cast<double>(), centroid3f.cast<double>(), radius_approx;

  std::vector<Eigen::Vector3d> data_points =
      getEigenCloud<pcl::PointXYZ>(cylinder_cloud);

  std::cout << "Initial guess " << x.transpose() << "\n";
  // Add the cost function for each data point
  for (int i = 0; i < data_points.size(); ++i) {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<CylinderCostFunctor, 1, 7>(
            new CylinderCostFunctor(data_points, i));
    problem.AddResidualBlock(cost_function, nullptr, x.data());
  }

  // Set up the solver options
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;

  // Solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  std::cout << "Solution: " << x.transpose() << "\n";
  return 0;
}
