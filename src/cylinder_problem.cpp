#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Dense>
#include <cstdlib>
#include <iostream>
#include <vector>

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

std::vector<Eigen::Vector3d> generate_cylinder_points(
    int n, const Eigen::Vector3d& axis, const Eigen::Vector3d& center,
    double radius, double height) {
  // Choose two orthogonal vectors perpendicular to the axis
  Eigen::Vector3d u, v;
  if (axis[0] != 0 || axis[1] != 0) {
    u = Eigen::Vector3d(axis[1], -axis[0], 0).normalized();
  } else {
    u = Eigen::Vector3d(0, axis[2], -axis[1]).normalized();
  }
  v = axis.cross(u);
  // Generate n points on the surface of the cylinder
  std::vector<Eigen::Vector3d> points;
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
    points.push_back(point);
  }

  return points;
}

int main() {
  Eigen::Vector3d axis, center;
  double radius = 1.0;
  axis << 0, 0.707, 0.707;
  center << 10, 20, 0;
  std::vector<Eigen::Vector3d> data_points =
      generate_cylinder_points(10000, axis, center, radius, 4.);
  for (int i = 0; i < data_points.size(); ++i) {
    std::cout << "point (" << i << "): " << data_points[i].transpose() << "\n";
  }

  // Set up the CERES problem
  ceres::Problem problem;
  Eigen::VectorXd x(7);

  double x_noise = (static_cast<double>(rand()) / RAND_MAX - 0.5);
  double y_noise = (static_cast<double>(rand()) / RAND_MAX - 0.5);
  double z_noise = (static_cast<double>(rand()) / RAND_MAX - 0.5);

  // x << axis,center,radius;
  x << axis(0) + x_noise, axis(1) + y_noise, axis(2) + z_noise,
      center(0) + x_noise, center(1) + y_noise, center(2) + z_noise,
      radius + x_noise;

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