#include <ceres/ceres.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <iostream>
#include <vector>

struct CostFunctor {
  CostFunctor(const std::vector<Eigen::Vector2d>& data_points, int i)
      : data_points(data_points), i(i) {}

  template <typename T>
  bool operator()(const T* const m, const T* const b, T* residual) const {
    residual[0] = T(data_points[i](1)) - (*m) * T(data_points[i](0)) - (*b);
    return true;
  }

  const std::vector<Eigen::Vector2d>& data_points;
  int i;
};

int main() {
  // Generate some random data points
  std::vector<Eigen::Vector2d> data_points;
  double m = 0.5;
  double b = 2.0;
  for (int i = 0; i < 100; ++i) {
    double x = (double)i / 100.0;
    double y = m * x + b + ((double)rand() / RAND_MAX - 0.5) * 0.2;
    data_points.push_back(Eigen::Vector2d(x, y));
  }

  // Set up the CERES problem
  ceres::Problem problem;

  // Add the cost function for each data point
  for (int i = 0; i < data_points.size(); ++i) {
    ceres::CostFunction* cost_function =
        new ceres::AutoDiffCostFunction<CostFunctor, 1, 1, 1>(
            new CostFunctor(data_points, i));
    problem.AddResidualBlock(cost_function, nullptr, &m, &b);
  }

  // Set up the solver options
  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;

  // Solve the problem
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);

  // Output the optimized parameters
  std::cout << "Initial m: " << 0.0 << ", b: " << 0.0 << std::endl;
  std::cout << "Final   m: " << m << ", b: " << b << std::endl;

  return 0;
}

