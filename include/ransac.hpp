#ifndef RANSAC_HPP
#define RANSAC_HPP
#include <pcl/point_cloud.h>

#include <Eigen/Core>
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

#endif  // RANSAC_HPP
