#include <pcl/features/normal_3d.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/segmentation/region_growing.h>
#include <pcl/visualization/pcl_visualizer.h>

#include "types.hpp"

template <typename PointT>
inline std::vector<PointCloudPtr<PointT>> euclidean_clustering(
    const PointCloudPtr<PointT> cloud, double cluster_tolerance,
    int min_cluster_size, int max_cluster_size) {
  std::vector<pcl::PointIndices> cluster_indices;
  typename pcl::search::KdTree<PointT>::Ptr tree(
      new pcl::search::KdTree<PointT>);
  tree->setInputCloud(cloud);

  pcl::EuclideanClusterExtraction<PointT> ec;
  ec.setClusterTolerance(cluster_tolerance);
  ec.setMinClusterSize(min_cluster_size);
  ec.setMaxClusterSize(max_cluster_size);
  ec.setSearchMethod(tree);
  ec.setInputCloud(cloud);
  ec.extract(cluster_indices);

  std::vector<PointCloudPtr<PointT>> clusters;
  for (const auto& indices : cluster_indices) {
    PointCloudPtr<PointT> cluster(new typename pcl::PointCloud<PointT>);
    for (const auto& index : indices.indices) {
      cluster->push_back((*cloud)[index]);
    }
    cluster->width = cluster->size();
    cluster->height = 1;
    cluster->is_dense = true;
    clusters.push_back(cluster);
  }

  return clusters;
}

template <typename PointT>
inline std::vector<PointCloudPtr<PointT>> region_growing_clustering(
    const PointCloudPtr<PointT> cloud, float normal_radius,
    float curvature_threshold, int min_cluster_size, int max_cluster_size) {
  // Create the RegionGrowing object
  pcl::RegionGrowing<PointT, pcl::Normal> reg;

  // Set the input cloud and its normals
  reg.setInputCloud(cloud);

  // Compute surface normals
  typename pcl::NormalEstimation<PointT, pcl::Normal> ne;
  typename pcl::search::KdTree<PointT>::Ptr tree(
      new pcl::search::KdTree<PointT>);
  ne.setSearchMethod(tree);
  ne.setInputCloud(cloud);
  pcl::PointCloud<pcl::Normal>::Ptr normals(new pcl::PointCloud<pcl::Normal>);
  ne.setRadiusSearch(normal_radius);
  ne.compute(*normals);
  reg.setInputNormals(normals);

  // Set the region growing parameters
  reg.setMinClusterSize(min_cluster_size);
  reg.setMaxClusterSize(max_cluster_size);
  reg.setSearchMethod(tree);
  reg.setNumberOfNeighbours(
      30);  // Number of neighbors to be considered for growing.
  reg.setSmoothnessThreshold(
      3.0 / 180.0 *
      M_PI);  // Set the angle threshold for smoothness constraints
  reg.setCurvatureThreshold(
      curvature_threshold);  // Set the curvature threshold

  // Perform region growing clustering
  std::vector<pcl::PointIndices> clusters;
  reg.extract(clusters);

  // Extract the clusters and store them in a vector of PointClouds
  std::vector<PointCloudPtr<PointT>> clustered_clouds;
  for (const auto& cluster_indices : clusters) {
    PointCloudPtr<PointT> cluster(new pcl::PointCloud<PointT>);
    for (const auto& idx : cluster_indices.indices) {
      cluster->push_back((*cloud)[idx]);
    }
    clustered_clouds.push_back(cluster);
  }

  return clustered_clouds;
}
