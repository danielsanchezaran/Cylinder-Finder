#ifndef RANSAC_HPP
#define RANSAC_HPP
#include <pcl/point_cloud.h>

#include <Eigen/Core>
#include <random>
#include <vector>


template <typename PointT>
inline const Eigen::MatrixXd pcl_to_eigen(
    typename pcl::PointCloud<PointT>::Ptr& cloud) {
  Eigen::MatrixXd out_cloud(3, cloud->size());
  int col = 0;
  for (const auto& p : *cloud) {
    Eigen::Vector3f q = p.getVector3fMap();
    out_cloud.col(col++) = q.cast<double>();
  }
  return out_cloud;
}

inline const Eigen::Vector3d circle_ransac(
    const Eigen::MatrixXd& data_points, int max_iterations,
    double threshold_distance, std::vector<int>& inlier_indices,
    const double acceptable_inlier_ratio = 0.95) {
  int num_points =
      data_points.cols();  // Assuming each column is a 2D data point

  if (num_points < 3) {
    throw std::invalid_argument("At least 3 data points are required.");
  }

  std::random_device rd;
  std::mt19937 generator(rd());

  Eigen::Vector3d best_circle;  // (center_x, center_y, radius)
  int best_inliers_count = 0;

  for (int iteration = 0; iteration < max_iterations; iteration++) {
    // Randomly select 3 data points to form a circle
    std::uniform_int_distribution<int> distribution(0, num_points - 1);
    int idx1 = distribution(generator);
    int idx2 = distribution(generator);
    int idx3 = distribution(generator);

    Eigen::Vector2d p1 = data_points.col(idx1);
    Eigen::Vector2d p2 = data_points.col(idx2);
    Eigen::Vector2d p3 = data_points.col(idx3);

    // Calculate the circumcenter of the triangle formed by the selected points
    double x1 = p1[0];
    double y1 = p1[1];
    double x2 = p2[0];
    double y2 = p2[1];
    double x3 = p3[0];
    double y3 = p3[1];

    double A = x1 * (y2 - y3) - y1 * (x2 - x3) + x2 * y3 - x3 * y2;
    if (std::abs(A) < 1e-9) continue;

    double B = (x1 * x1 + y1 * y1) * (y3 - y2) +
               (x2 * x2 + y2 * y2) * (y1 - y3) +
               (x3 * x3 + y3 * y3) * (y2 - y1);
    double C = (x1 * x1 + y1 * y1) * (x2 - x3) +
               (x2 * x2 + y2 * y2) * (x3 - x1) +
               (x3 * x3 + y3 * y3) * (x1 - x2);

    double D = (x1 * x1 + y1 * y1) * (x3 * y2 - x2 * y3) +
               (x2 * x2 + y2 * y2) * (x1 * y3 - x3 * y1) +
               (x3 * x3 + y3 * y3) * (x2 * y1 - x1 * y2);

    double xc = -B / (2 * A);
    double yc = -C / (2 * A);
    double radius = std::sqrt((B * B + C * C - 4 * A * D) / (4 * A * A));
    // Calculate the circle radius as the distance between the circumcenter and
    // any of the points

    // Count inliers (data points close enough to the circle)
    Eigen::Vector2d center(xc, yc);
    std::vector<int> inliers;
    for (int i = 0; i < num_points; i++) {
      Eigen::Vector2d p = data_points.col(i);
      double distance = (p - center).norm();

      if (std::abs(distance - radius) < threshold_distance) {
        inliers.push_back(i);
      }
    }

    int inliers_count = inliers.size();

    // Check if this circle model is better than the current best model
    if (inliers_count > best_inliers_count) {
      best_inliers_count = inliers_count;
      best_circle << center, radius;
      inlier_indices = inliers;
    }
    if (best_inliers_count > num_points * acceptable_inlier_ratio) break;
  }

  return best_circle;
}


#endif  // RANSAC_HPP
