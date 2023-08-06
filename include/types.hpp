#ifndef TYPES_HPP_
#define TYPES_HPP_

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

template <typename PointT>
using PointCloudPtr = typename pcl::PointCloud<PointT>::Ptr;

#endif // TYPES_HPP_