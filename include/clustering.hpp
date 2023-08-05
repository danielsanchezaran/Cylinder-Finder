#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>

template <typename PointT>
inline std::vector<typename pcl::PointCloud<PointT>::Ptr> euclidean_clustering(
    const typename pcl::PointCloud<PointT>::Ptr cloud, double cluster_tolerance,
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

  std::vector<typename pcl::PointCloud<PointT>::Ptr> clusters;
  for (const auto& indices : cluster_indices) {
    typename pcl::PointCloud<PointT>::Ptr cluster(
        new typename pcl::PointCloud<PointT>);
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
