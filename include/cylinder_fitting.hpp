#ifndef CYLINDER_FITTING_H
#define CYLINDER_FITTING_H

#include <ceres/ceres.h>
#include <pcl-1.10/pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>
#include <vector>

/**
 * CylinderCostFunctor: A functor used as a cost function in Ceres optimization
 * to compute the residual distance between a data point and a cylinder's
 * surface.
 * @param data_points A vector of Eigen 3D vectors representing the data points.
 * @param i Index of the data point in the vector.
 */
struct CylinderCostFunctor {
  CylinderCostFunctor(const std::vector<Eigen::Vector3d>& data_points, int i)
      : data_points(data_points), i(i) {}
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    Eigen::Matrix<T, 3, 1> axis(x[0], x[1], x[2]);
    Eigen::Matrix<T, 3, 1> center(x[3], x[4], x[5]);
    T radius = x[6];
    // Compute the distance from the data point to the axis
    Eigen::Matrix<T, 3, 1> p(data_points[i].cast<T>());
    Eigen::Matrix<T, 3, 1> v(p - center);
    T mag = v.dot(axis.normalized());
    Eigen::Matrix<T, 3, 1> point_in_axis(center + mag * axis);
    T d = (p - point_in_axis).norm();

    // Compute the residual
    residual[0] = d * d - (radius * radius);
    return true;
  }
  const std::vector<Eigen::Vector3d>& data_points;
  int i;
};

/**
 * project_points_perpendicular_to_axis: Projects a point cloud onto a plane
 * perpendicular to the given axis.
 * @param cloud_in Input point cloud to be projected.
 * @param axis The axis used for the projection.
 * @return A new point cloud representing the projection result.
 */
template <typename PointT>
inline pcl::PointCloud<PointT> project_points_perpendicular_to_axis(
    const pcl::PointCloud<PointT>& cloud_in, const Eigen::Vector3f& axis) {
  // Project the points
  pcl::PointCloud<PointT> cloud_out;
  for (const auto& p : cloud_in) {
    Eigen::Vector3f q = p.getVector3fMap();
    q = q - axis.normalized() * q.dot(axis.normalized());
    cloud_out.push_back(PointT(q.x(), q.y(), q.z()));
  }

  return cloud_out;
}

/**
 * computeCentroid: Computes the centroid of a point cloud and returns it as a
 * 4D vector.
 * @param cloud The input point cloud.
 * @return A 4D vector representing the centroid (x, y, z, 1.0).
 */
template <typename PointT>
inline Eigen::Vector4f computeCentroid(
    const typename pcl::PointCloud<PointT>::Ptr& cloud) {
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);
  return centroid;
}

/**
 * getEigenCloud: Converts a PCL point cloud to a vector of Eigen 3D vectors.
 * @param cloud Input PCL point cloud.
 * @return A vector of Eigen 3D vectors representing the point cloud.
 */
template <typename PointT>
inline std::vector<Eigen::Vector3d> getEigenCloud(
    typename pcl::PointCloud<PointT>::Ptr cloud) {
  std::vector<Eigen::Vector3d> cloud_eigen;
  for (const auto& point : cloud->points) {
    Eigen::Vector3f p = point.getVector3fMap();
    cloud_eigen.push_back(p.cast<double>());
  }
  return cloud_eigen;
}

/**
 * compute_cylinder_radius: Computes the mean distance from a point cloud to a
 * given axis, representing the cylinder's approximate radius.
 * @param cloud The input point cloud.
 * @param axis The axis used for computing the radius.
 * @param origin The origin point on the axis.
 * @return The computed cylinder radius.
 */
template <typename PointT>
inline double compute_cylinder_radius(
    typename pcl::PointCloud<PointT>::Ptr cloud, const Eigen::Vector3f& axis,
    const Eigen::Vector3f& origin) {
  double variance = 0.0;
  double mean_distance = 0.0;

  for (const auto& point : cloud->points) {
    Eigen::Vector3f p = point.getVector3fMap();
    Eigen::Vector3f v = p - origin;
    Eigen::Vector3f proj =
        origin + v.dot(axis.normalized()) * axis.normalized();
    double distance = (p - proj).norm();
    mean_distance += distance;
  }

  mean_distance /= cloud->size();
  return mean_distance;
}

/**
 * generate_cylinder_points: Generates points on the surface of a cylinder with
 * specified axis, center, radius, and height.
 * @param n The number of points to be generated.
 * @param axis The axis of the cylinder.
 * @param center The center point of the cylinder.
 * @param radius The radius of the cylinder.
 * @param height The height of the cylinder.
 * @param cylinder_cloud The output point cloud to store the generated points.
 */
template <typename PointT>
inline void generate_cylinder_points(
    int n, const Eigen::Vector3d& axis, const Eigen::Vector3d& center,
    double radius, double height,
    typename pcl::PointCloud<PointT>::Ptr& cylinder_cloud) {
  // Choose two orthogonal vectors perpendicular to the axis
  Eigen::Vector3d u, v;
  if (axis[0] != 0 || axis[1] != 0) {
    u = Eigen::Vector3d(axis[1], -axis[0], 0).normalized();
  } else {
    u = Eigen::Vector3d(0, axis[2], -axis[1]).normalized();
  }
  v = axis.cross(u);

  // Generate n points on the surface of the cylinder
  cylinder_cloud->points.resize(n);
  for (int i = 0; i < n; ++i) {
    double theta = (static_cast<double>(i) / (double)n) * 2.0 * M_PI;
    double x_noise =
        (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.1 * radius;
    double y_noise =
        (static_cast<double>(rand()) / RAND_MAX - 0.5) * 0.1 * radius;
    double z_noise =
        (static_cast<double>(rand()) / RAND_MAX - 0.5) * 1.0 * height;

    Eigen::Vector3d relative_position;
    relative_position << std::cos(theta) * (radius + x_noise) * u +
                             std::sin(theta) * (radius + y_noise) * v +
                             z_noise * axis.normalized();
    Eigen::Vector3d point = center + relative_position;

    // Add the generated point to the output point cloud
    PointT pcl_point;
    pcl_point.x = point.x();
    pcl_point.y = point.y();
    pcl_point.z = point.z();
    cylinder_cloud->points[i] = pcl_point;
  }
}

/**
 * Estimates the parameters of a cylinder-like structure from a given point
 cloud.
   @param cylinder_cloud A shared pointer to the input point cloud containing
 the points of the cylinder.
   @param main_axis A 3D vector representing the estimated main axis of the
 cylinder.
   @param centroid3f A 3D vector representing the estimated centroid of the
 cylinder.
   @param radius_approx The estimated radius of the cylinder.
*/
template <typename PointT>
inline void estimate_cylinder_parameters(
    typename pcl::PointCloud<PointT>::Ptr& cylinder_cloud,
    Eigen::Vector3f& main_axis, Eigen::Vector3f& centroid3f,
    double& radius_approx) {
  auto cylinder_centroid = computeCentroid<PointT>(cylinder_cloud);
  centroid3f << cylinder_centroid(0), cylinder_centroid(1),
      cylinder_centroid(2);

  pcl::PCA<PointT> pca;
  pca.setInputCloud(cylinder_cloud);
  main_axis = pca.getEigenVectors().col(2);
  radius_approx =
      compute_cylinder_radius<PointT>(cylinder_cloud, main_axis, centroid3f);
  auto projected_points =
      project_points_perpendicular_to_axis<PointT>(*cylinder_cloud, main_axis);
}

/**
 * find_cylinder_model: Finds the parameters (axis, centroid, and radius) of a
 * cylinder model in a given point cloud using Ceres optimization.
 * @param cylinder_cloud The input point cloud containing the cylinder points.
 * @return A 7D vector containing the cylinder parameters: main_axis (3D
 * vector), centroid (3D vector), and radius (scalar).
 */
template <typename PointT>
inline const Eigen::VectorXd find_cylinder_model(
    typename pcl::PointCloud<PointT>::Ptr& cylinder_cloud) {
  double radius_approx;
  Eigen::Vector3f centroid3f;
  Eigen::Vector3f main_axis;
  estimate_cylinder_parameters<PointT>(cylinder_cloud, main_axis, centroid3f,
                                       radius_approx);
  ceres::Problem problem;
  Eigen::VectorXd x(7);

  x << main_axis.cast<double>(), centroid3f.cast<double>(), radius_approx;

  std::vector<Eigen::Vector3d> data_points =
      getEigenCloud<PointT>(cylinder_cloud);

  // Add the cost function for each data point
  for (int i = 0; i < data_points.size(); ++i) {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<CylinderCostFunctor, 1, 7>(
            new CylinderCostFunctor(data_points, i));
    problem.AddResidualBlock(cost_function, nullptr, x.data());
  }

  // Set up the solver options
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = false;

  // Solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  return x;
}

template <typename PointT>
inline const Eigen::VectorXd find_cylinder_projection_ransac(
    typename pcl::PointCloud<PointT>::Ptr& cylinder_cloud) {
  double radius_approx;
  Eigen::Vector3f centroid3f;
  Eigen::Vector3f main_axis;
  estimate_cylinder_parameters<PointT>(cylinder_cloud, main_axis, centroid3f,
                                       radius_approx);
}

/**
 * projection_of_point_in_axis: Computes the projection of a point onto a given
 * axis.
 * @param direction The direction vector of the axis.
 * @param line_point A point on the axis.
 * @param point The point to be projected.
 * @return The projection of the point onto the axis.
 */
inline double projection_of_point_in_axis(const Eigen::Vector3d& direction,
                                          const Eigen::Vector3d& line_point,
                                          const Eigen::Vector3d& point) {
  auto line_point_to_point = point - line_point;
  return line_point_to_point.dot(direction);
}

/**
 * adjust_cylinder_model_to_points: Adjusts the cylinder model parameters to fit
 * the points in the point cloud more accurately.
 * @param cylinder_cloud The input point cloud containing the cylinder points.
 * @param cylinder_coefficients The cylinder parameters: main_axis (3D vector),
 * centroid (3D vector), and radius (scalar).
 */
template <typename PointT>
inline void adjust_cylinder_model_to_points(
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

/**
 * filter_cylinder_inliers: Filters the inlier points of a cylinder model based
 * on a specified radius percentage.
 * @param cylinder_cloud The input point cloud containing the cylinder points.
 * @param cylinder_coefficients The cylinder parameters: main_axis (3D vector),
 * centroid (3D vector), and radius (scalar).
 * @param radius_percentage The percentage of radius used as the filtering
 * threshold.
 * @return A new point cloud containing the filtered inlier points.
 */
template <typename PointT>
inline typename pcl::PointCloud<PointT>::Ptr filter_cylinder_inliers(
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

#endif  // CYLINDER_FITTING_H
