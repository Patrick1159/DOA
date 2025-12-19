#ifndef DYNAMIC_OBSTACLE_LAYER_H_
#define DYNAMIC_OBSTACLE_LAYER_H_

#include <ros/ros.h>
#include <costmap_2d/layer.h>
#include <costmap_2d/costmap_layer.h>
#include <costmap_2d/GenericPluginConfig.h>
#include <dynamic_reconfigure/server.h>
#include <umap/TrackerDetails.h>
#include <mutex>
#include <vector>
#include <cmath>

namespace path_planning
{

class DynamicObstacleLayer : public costmap_2d::CostmapLayer
{
public:
  DynamicObstacleLayer();
  virtual ~DynamicObstacleLayer();

  virtual void onInitialize();
  virtual void updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y, double* max_x, double* max_y);
  virtual void updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j);
  
  virtual void matchSize();
  virtual void reset();

private:
  void reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level);
  void trackerCallback(const umap::TrackerDetails::ConstPtr& msg);
  
  // Helper to update costs for a single obstacle
  void processObstacle(const geometry_msgs::Point& pos, 
                       const geometry_msgs::Vector3& vel, 
                       const geometry_msgs::Vector3& dims,
                       double* min_x, double* min_y, double* max_x, double* max_y);

  dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig> *dsrv_;
  ros::Subscriber tracker_sub_;
  
  std::mutex data_mutex_;
  umap::TrackerDetails::ConstPtr last_tracker_msg_;
  bool new_data_received_;

  // Timestamp grid for decay logic
  // Stores the last time a cell was actively updated by an obstacle
  std::vector<double> last_seen_timestamps_;
  
  // Time of the last update cycle, used for calculating decay delta
  double last_update_time_;

  // Parameters
  double decay_rate_;          // Exponential decay rate
  double keep_alive_time_;     // Time before decay starts
  double velocity_factor_;     // Factor to scale influence by velocity
  double max_prediction_time_; // Max time horizon for velocity projection
  double obstacle_radius_buffer_; // Extra buffer around the obstacle box
};

} // namespace path_planning

#endif
