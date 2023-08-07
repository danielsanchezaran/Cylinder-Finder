#include <pcl/visualization/pcl_visualizer.h>

#include <future>
#include <iostream>
#include <limits>
#include <string>

#include "clustering.hpp"
#include "cylinder_fitting.hpp"
#include "ransac.hpp"
#include "thread_pool.hpp"
#include "utils.hpp"

int main() {
  ThreadPool thread_pool(10);

  // Define the vector to hold the PointCloud pointers
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> cylinder_clouds;

  int n_point_clouds = 4;
  for (int i = 0; i < n_point_clouds; ++i) {
    pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(
        new pcl::PointCloud<pcl::PointXYZ>);
    cylinder_clouds.push_back(cloud);
  }

  // unsigned int seed = 210;
  std::random_device rd;
  std::mt19937 generator(rd());
  // std::mt19937 generator(seed);

  std::uniform_real_distribution<double> distribution_radius(0.05, 15.);
  std::uniform_real_distribution<double> distribution_ratio(1.1, 1.8);
  std::uniform_real_distribution<double> distribution_center(-30., 30.);
  std::uniform_int_distribution<int> distribution_points(50, 1000);
  std::vector<pcl::PointCloud<pcl::PointXYZ>::Ptr> clusters;
  pcl::PointCloud<pcl::PointXYZ>::Ptr combined_clouds(
      new pcl::PointCloud<pcl::PointXYZ>);
  {
    TimeIT t("Clustering");
    for (int i = 0; i < n_point_clouds; ++i) {
      double axis1 = distribution_center(generator);
      double axis2 = distribution_center(generator);
      double axis3 = distribution_center(generator);

      if (axis1 == 0.0 && axis2 == 0.0 && axis3 == 0.0) axis1 = 1.0;
      Eigen::Vector3d axis(axis1, axis2, axis3);
      axis.normalize();

      double center_x = distribution_center(generator);
      double center_y = distribution_center(generator);
      double center_z = distribution_center(generator);
      Eigen::Vector3d center(center_x, center_y, center_z);

      double radius = distribution_radius(generator);
      double height = distribution_ratio(generator) * radius;
      // int n_points = distribution_points(generator);
      int n_points = 1000;

      thread_pool.enqueue([=] {
        generate_cylinder_points<pcl::PointXYZ>(n_points, axis, center, radius,
                                                height, cylinder_clouds[i]);
      });
    }
    thread_pool.waitUntilDone();

    for (int i = 0; i < n_point_clouds; ++i)
      *combined_clouds += *cylinder_clouds[i];
    // Set the clustering parameters
    double cluster_tolerance = 1.5;
    int min_cluster_size = 50;
    int max_cluster_size = 1000;

    // Call the clustering function
    // clusters = euclidean_clustering<pcl::PointXYZ>(
    //     combined_clouds, cluster_tolerance, min_cluster_size,
    //     max_cluster_size);
    clusters = region_growing_clustering<pcl::PointXYZ>(combined_clouds, 0.25,
                                                        0.25, 50, 1500);
  }

  // Visualize the original point cloud
  pcl::visualization::PCLVisualizer::Ptr viewer(
      new pcl::visualization::PCLVisualizer("Original Cloud"));
  viewer->addPointCloud<pcl::PointXYZ>(combined_clouds, "cloud");
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
