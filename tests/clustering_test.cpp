#include "clustering.hpp"

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>

#include <thread>

pcl::PointCloud<pcl::PointXYZ>::Ptr generateTestPointCloud(int num_points) {
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

  // Define the bounding box size
  double min_x = -10.0;
  double max_x = 10.0;
  double min_y = -10.0;
  double max_y = 10.0;
  double min_z = 0.0;
  double max_z = 5.0;

  // Generate random points within the bounding box
  for (int i = 0; i < num_points; ++i) {
    pcl::PointXYZ point;
    point.x = static_cast<float>(min_x + static_cast<double>(rand()) /
                                             static_cast<double>(RAND_MAX) *
                                             (max_x - min_x));
    point.y = static_cast<float>(min_y + static_cast<double>(rand()) /
                                             static_cast<double>(RAND_MAX) *
                                             (max_y - min_y));
    point.z = static_cast<float>(min_z + static_cast<double>(rand()) /
                                             static_cast<double>(RAND_MAX) *
                                             (max_z - min_z));
    cloud->push_back(point);

    if (i == num_points / 2) {
      min_x += 20; 
      max_x += 20; 
      min_y += 20; 
      max_y += 20; 
      min_z += 20; 
      max_z += 20; 
    }
  }

  cloud->width = cloud->size();
  cloud->height = 1;
  cloud->is_dense = true;

  return cloud;
}

void testEuclideanClustering() {
  // Generate a synthetic test point cloud with 1000 random points
  pcl::PointCloud<pcl::PointXYZ>::Ptr cloud = generateTestPointCloud(1000);

  // Set the clustering parameters
  double cluster_tolerance = 5;
  int min_cluster_size = 20;
  int max_cluster_size = 2500;

  // Call the clustering function
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters =
      euclidean_clustering<pcl::PointXYZ>(cloud, cluster_tolerance,
                                          min_cluster_size, max_cluster_size);

  // Visualize the original point cloud
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Original Cloud"));
  viewer->addPointCloud<pcl::PointXYZ>(cloud, "cloud");
  viewer->setPointCloudRenderingProperties(
      pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 1, "cloud");

    std::cout << "Clusters " << clusters.size() << "\n";
  // Visualize the clusters in different colors
  for (size_t i = 0; i < clusters.size(); ++i) {
    pcl::visualization::PointCloudColorHandlerRandom<pcl::PointXYZ>
        color_handler(clusters[i]);
    viewer->addPointCloud<pcl::PointXYZ>(clusters[i], color_handler,
                                         "cluster_" + std::to_string(i));
    viewer->setPointCloudRenderingProperties(
        pcl::visualization::PCL_VISUALIZER_POINT_SIZE, 2,
        "cluster_" + std::to_string(i));
  }

  // Display the point cloud and clusters
  while (!viewer->wasStopped()) {
    viewer->spinOnce(100);
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
  }
}

int main() {
  testEuclideanClustering();
  return 0;
}
