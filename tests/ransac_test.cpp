#include "ransac.hpp"

#include <gtest/gtest.h>

#include <Eigen/Dense>
#include <cmath>

// Include the circle_ransac function here

TEST(CircleRANSACTest, BasicTest) {
  const double center_x = -2.0;
  const double center_y = -10.0;
  const double radius = 5.82;
  const int num_points = 100;
  double noise = 0.1;

  Eigen::MatrixXd data_points(2, num_points);
  for (int i = 0; i < num_points; i++) {
    double angle =
        2.0 * M_PI * static_cast<double>(i) / static_cast<double>(num_points);
    double x = center_x + radius * std::cos(angle);
    double y = center_y + radius * std::sin(angle);

    // Add some random noise to the points to simulate noisy data
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<double> noise_distribution(-noise, noise);
    x += noise_distribution(generator);
    y += noise_distribution(generator);

    data_points.col(i) << x, y;
  }

  std::vector<int> inlier_indices;
  Eigen::Vector3d result =
      circle_ransac(data_points, 1000, noise * std::sqrt(2), inlier_indices);

  // The correct center should be centerx, centery, and the radius should be 2.0
  // (approximately).
  EXPECT_NEAR(result[0], center_x, noise * std::sqrt(2));
  EXPECT_NEAR(result[1], center_y, noise * std::sqrt(2));
  EXPECT_NEAR(result[2], radius, noise * std::sqrt(2));

  // Verify that the inliers correspond to the points that are close enough to
  // the circle.
  Eigen::Vector2d ransac_center(result[0],result[1]);
  for (int index : inlier_indices) {
    Eigen::Vector2d p = data_points.col(index);
    double distance = (ransac_center - p).norm();
    EXPECT_NEAR(std::abs(distance - result[2]), 0.0, 0.15);
  }
}

TEST(CircleRANSACTest, FewDataPointsTest) {
  Eigen::MatrixXd data_points(2, 2);
  data_points << 0.0, 1.0, 0.0, 1.0;
  std::vector<int> inlier_indices;
  // The function should throw an exception for less than 3 data points.
  EXPECT_THROW(circle_ransac(data_points, 1000, 0.1, inlier_indices),
               std::invalid_argument);
}

// Add more tests if needed

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
