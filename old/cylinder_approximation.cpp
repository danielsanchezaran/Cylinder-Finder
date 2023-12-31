#include <ceres/ceres.h>
#include <pcl-1.10/pcl/common/centroid.h>
#include <pcl/common/pca.h>
#include <pcl/features/feature.h>
#include <pcl/features/normal_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <Eigen/Core>

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

template <typename PointT>
pcl::PointCloud<PointT> project_points_perpendicular_to_axis(
    const pcl::PointCloud<PointT>& cloud_in, const Eigen::Vector3f& axis) {
  // Compute the projection matrix
  Eigen::Matrix3f P = Eigen::Matrix3f::Identity() - axis * axis.transpose();

  // Project the points
  pcl::PointCloud<PointT> cloud_out;
  for (const auto& p : cloud_in) {
    Eigen::Vector3f q = P * p.getVector3fMap();
    cloud_out.push_back(PointT(q.x(), q.y(), q.z()));
  }

  return cloud_out;
}

template <typename PointT>
Eigen::Vector4f computeCentroid(
    const typename pcl::PointCloud<PointT>::Ptr& cloud) {
  Eigen::Vector4f centroid;
  pcl::compute3DCentroid(*cloud, centroid);
  return centroid;
}

template <typename PointT>
std::vector<Eigen::Vector3d> get_eigen_cloud(
    typename pcl::PointCloud<PointT>::Ptr cloud) {
  std::vector<Eigen::Vector3d> cloud_eigen;
  for (const auto& point : cloud->points) {
    Eigen::Vector3f p = point.getVector3fMap();
    cloud_eigen.push_back(p.cast<double>());
  }
  return cloud_eigen;
}

template <typename PointT>
double compute_cylinder_radius(typename pcl::PointCloud<PointT>::Ptr cloud,
                               const Eigen::Vector3f& axis,
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

template <typename PointT>
void generate_cylinder_points(
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
        (static_cast<double>(rand()) / RAND_MAX - 0.5) * 2.0 * height;

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

  std::vector<Eigen::Vector3d> data_points = get_eigen_cloud<pcl::PointXYZ>(cylinder_cloud);

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