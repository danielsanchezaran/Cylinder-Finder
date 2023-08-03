#ifndef COST_FUNCTORS_HPP
#define COST_FUNCTORS_HPP

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
 * @brief Functor to calculate the residual for Ceres optimization.
 *
 * @param data_points A 2xN MatrixXd representing the data points.
 * @param i The index of the data point being processed.
 */
struct CircleCostFunctor {
  CircleCostFunctor(const Eigen::MatrixXd& data_points, int i)
      : data_points(data_points), i(i) {}
  template <typename T>
  bool operator()(const T* const x, T* residual) const {
    Eigen::Matrix<T, 2, 1> center(x[0], x[1]);
    T radius = x[2];
    // Compute the distance from the data point to the center
    Eigen::Matrix<T, 2, 1> p(data_points.col(i).cast<T>());
    Eigen::Matrix<T, 2, 1> v(p - center);
    T r = v.norm();

    // Compute the residual
    // residual[0] = r * r - (radius * radius);
    residual[0] = r - radius;

    return true;
  }
  const Eigen::MatrixXd& data_points;
  int i;
};

#endif // COST_FUNCTORS_HPP