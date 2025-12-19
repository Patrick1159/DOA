#include <path_planning/dynamic_obstacle_layer.h>
#include <pluginlib/class_list_macros.h>
#include <geometry_msgs/Point.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <tf2/LinearMath/Quaternion.h>

PLUGINLIB_EXPORT_CLASS(path_planning::DynamicObstacleLayer, costmap_2d::Layer)

using costmap_2d::LETHAL_OBSTACLE;
using costmap_2d::NO_INFORMATION;
using costmap_2d::FREE_SPACE;

namespace path_planning
{

DynamicObstacleLayer::DynamicObstacleLayer() 
  : last_update_time_(0.0)
{
  costmap_ = NULL; 
}

DynamicObstacleLayer::~DynamicObstacleLayer() {}

void DynamicObstacleLayer::onInitialize()
{
  ros::NodeHandle nh("~/" + name_);
  current_ = true;
  default_value_ = FREE_SPACE; // Default to free space so we can decay to it
  
  matchSize();
  
  dsrv_ = new dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>(nh);
  dynamic_reconfigure::Server<costmap_2d::GenericPluginConfig>::CallbackType cb = boost::bind(
      &DynamicObstacleLayer::reconfigureCB, this, _1, _2);
  dsrv_->setCallback(cb);

  // Parameters
  nh.param("decay_rate", decay_rate_, 2.0); 
  nh.param("keep_alive_time", keep_alive_time_, 0.2); 
  nh.param("velocity_factor", velocity_factor_, 0.5);
  nh.param("max_prediction_time", max_prediction_time_, 1.5);
  nh.param("obstacle_radius_buffer", obstacle_radius_buffer_, 0.05);

  tracker_sub_ = nh.subscribe("/rematch/tracker_details", 1, &DynamicObstacleLayer::trackerCallback, this);
  
  new_data_received_ = false;
  last_update_time_ = ros::Time::now().toSec();
}

void DynamicObstacleLayer::matchSize()
{
  CostmapLayer::matchSize();
  last_seen_timestamps_.resize(getSizeInCellsX() * getSizeInCellsY(), 0.0);
}

void DynamicObstacleLayer::reconfigureCB(costmap_2d::GenericPluginConfig &config, uint32_t level)
{
  enabled_ = config.enabled;
}

void DynamicObstacleLayer::trackerCallback(const umap::TrackerDetails::ConstPtr& msg)
{
  std::lock_guard<std::mutex> lock(data_mutex_);
  last_tracker_msg_ = msg;
  new_data_received_ = true;
  // ROS_DEBUG_THROTTLE(2.0, "DynamicObstacleLayer: Received tracker data with %zu obstacles", msg->tracker_ids.size());
  ROS_INFO_THROTTLE(2.0, "DynamicObstacleLayer: Received tracker data with %zu obstacles", msg->tracker_ids.size());
}

void DynamicObstacleLayer::updateBounds(double robot_x, double robot_y, double robot_yaw, double* min_x, double* min_y, double* max_x, double* max_y)
{
  if (!enabled_) 
  {
      ROS_WARN_THROTTLE(5.0, "DynamicObstacleLayer: Layer is disabled.");
      return;
  }

  // Sync with master costmap origin for rolling window support
  if (layered_costmap_->isRolling())
  {
      double master_origin_x = layered_costmap_->getCostmap()->getOriginX();
      double master_origin_y = layered_costmap_->getCostmap()->getOriginY();
      
      if (std::abs(master_origin_x - getOriginX()) > 1e-4 || std::abs(master_origin_y - getOriginY()) > 1e-4)
      {
          ROS_INFO_THROTTLE(1.0, "DynamicObstacleLayer: Origin changed. Resetting map to sync with master.");
          updateOrigin(master_origin_x, master_origin_y);
          // Reset map to clear old data that is now misaligned
          // Note: This clears decay history, but ensures active obstacles are placed correctly.
          // A better implementation would shift the data.
          reset(); 
      }
  }

  double current_time = ros::Time::now().toSec();
  double dt = current_time - last_update_time_;
  last_update_time_ = current_time;

  // 1. Decay existing costs
  unsigned int size_x = getSizeInCellsX();
  unsigned int size_y = getSizeInCellsY();
  
  // Only iterate if we have something to decay (optimization possible, but full iteration is safe for local map)
  for (unsigned int i = 0; i < size_x * size_y; ++i)
  {
    unsigned char cost = costmap_[i];
    if (cost != NO_INFORMATION && cost != FREE_SPACE)
    {
      double time_since_seen = current_time - last_seen_timestamps_[i];
      if (time_since_seen > keep_alive_time_)
      {
        double factor = exp(-decay_rate_ * dt);
        // Decay logic: map 0-254. 
        // If LETHAL (254), we might want to drop it to non-lethal quickly.
        double new_cost_d = (double)cost * factor;
        
        if (new_cost_d < 1.0) 
        {
          costmap_[i] = FREE_SPACE;
        }
        else
        {
          costmap_[i] = (unsigned char)new_cost_d;
        }
        
        // Update bounds
        unsigned int mx, my;
        indexToCells(i, mx, my);
        double wx, wy;
        mapToWorld(mx, my, wx, wy);
        *min_x = std::min(*min_x, wx);
        *min_y = std::min(*min_y, wy);
        *max_x = std::max(*max_x, wx);
        *max_y = std::max(*max_y, wy);
      }
    }
  }

  // 2. Process new tracker data
  umap::TrackerDetails::ConstPtr msg;
  {
    std::lock_guard<std::mutex> lock(data_mutex_);
    if (last_tracker_msg_)
    {
        msg = last_tracker_msg_;
    }
  }

  if (!msg)
  {
      ROS_WARN_THROTTLE(5.0, "DynamicObstacleLayer: No tracker data received yet.");
      return;
  }

  if (msg)
  {
    // Transform to costmap frame if needed
    // Assuming costmap frame is "odom" or "map" and msg is in "odom" or "map"
    // We need to transform each point.
    
    geometry_msgs::TransformStamped transform;
    try
    {
      transform = tf_->lookupTransform(layered_costmap_->getGlobalFrameID(), msg->header.frame_id, ros::Time(0));
    }
    catch (tf2::TransformException &ex)
    {
      ROS_WARN_THROTTLE(1.0, "DynamicObstacleLayer: TF Error: %s", ex.what());
      return;
    }

    if (msg->tracker_ids.empty())
    {
        ROS_INFO_THROTTLE(2.0, "DynamicObstacleLayer: Tracker message is empty.");
    }
    else
    {
        ROS_INFO_THROTTLE(2.0, "DynamicObstacleLayer: Processing %zu obstacles. Frame: %s -> %s", 
            msg->tracker_ids.size(), msg->header.frame_id.c_str(), layered_costmap_->getGlobalFrameID().c_str());
    }

    for (size_t i = 0; i < msg->tracker_ids.size(); ++i)
    {
      geometry_msgs::PointStamped pt_in, pt_out;
      pt_in.header = msg->header;
      pt_in.point = msg->positions[i];
      
      geometry_msgs::Vector3Stamped vel_in, vel_out;
      vel_in.header = msg->header;
      vel_in.vector = msg->velocities[i];

      // Transform point
      tf2::doTransform(pt_in, pt_out, transform);
      
      // Transform velocity (rotation only)
      // tf2::doTransform for Vector3Stamped handles rotation
      tf2::doTransform(vel_in, vel_out, transform);

      processObstacle(pt_out.point, vel_out.vector, msg->box_dimensions[i], min_x, min_y, max_x, max_y);
    }
  }
}

void DynamicObstacleLayer::processObstacle(const geometry_msgs::Point& pos, 
                                           const geometry_msgs::Vector3& vel, 
                                           const geometry_msgs::Vector3& dims,
                                           double* min_x, double* min_y, double* max_x, double* max_y)
{
    double cx = pos.x;
    double cy = pos.y;
    double vx = vel.x;
    double vy = vel.y;
    double v_mag = std::hypot(vx, vy);
    double yaw = std::atan2(vy, vx);

    // Obstacle dimensions (assume box aligned with velocity or axis? 
    // TrackerDetails usually gives axis aligned or oriented box. 
    // If box_dimensions is just size, we assume it's oriented with velocity or just a blob.
    // Let's assume it's a radius or small box.
    // Using max dimension for safety radius.
    double radius = std::max(dims.x, dims.y) / 2.0 + obstacle_radius_buffer_;
    
    // Calculate influence region
    double prediction_dist = std::min(v_mag * max_prediction_time_, 5.0); // Cap at 5m
    double influence_radius = radius + prediction_dist;

    // Bounding box for iteration
    double bx_min = cx - influence_radius;
    double bx_max = cx + influence_radius;
    double by_min = cy - influence_radius;
    double by_max = cy + influence_radius;

    // Clamp to map bounds in world coordinates to ensure worldToMap succeeds
    double origin_x = getOriginX();
    double origin_y = getOriginY();
    double map_end_x = origin_x + getSizeInCellsX() * getResolution();
    double map_end_y = origin_y + getSizeInCellsY() * getResolution();

    if (bx_min < origin_x) bx_min = origin_x;
    if (by_min < origin_y) by_min = origin_y;
    if (bx_max >= map_end_x) bx_max = map_end_x - 0.001;
    if (by_max >= map_end_y) by_max = map_end_y - 0.001;

    if (bx_min >= bx_max || by_min >= by_max) 
    {
        ROS_WARN_THROTTLE(1.0, "DynamicObstacleLayer: Obstacle out of bounds. Pos: (%.2f, %.2f), Map: [%.2f, %.2f] x [%.2f, %.2f]", 
            cx, cy, origin_x, map_end_x, origin_y, map_end_y);
        return;
    }

    unsigned int ui_min_i, ui_min_j, ui_max_i, ui_max_j;
    if (!worldToMap(bx_min, by_min, ui_min_i, ui_min_j)) return;
    if (!worldToMap(bx_max, by_max, ui_max_i, ui_max_j)) return;

    int min_i = (int)ui_min_i;
    int min_j = (int)ui_min_j;
    int max_i = (int)ui_max_i;
    int max_j = (int)ui_max_j;

    // ROS_INFO_THROTTLE(1.0, "DynamicObstacleLayer: Checking cells [%d, %d] to [%d, %d] for obstacle at (%.2f, %.2f)", 
    //     min_i, min_j, max_i, max_j, cx, cy);

    double current_time = ros::Time::now().toSec();

    bool updated_any = false;
    for (int j = min_j; j <= max_j; j++)
    {
        for (int i = min_i; i <= max_i; i++)
        {
            double wx, wy;
            mapToWorld(i, j, wx, wy);
            
            double dx = wx - cx;
            double dy = wy - cy;
            double dist_sq = dx*dx + dy*dy;

            // Check if inside actual obstacle (LETHAL)
            if (dist_sq < radius * radius)
            {
                unsigned int idx = getIndex(i, j);
                costmap_[idx] = LETHAL_OBSTACLE;
                last_seen_timestamps_[idx] = current_time;
                
                *min_x = std::min(*min_x, wx);
                *min_y = std::min(*min_y, wy);
                *max_x = std::max(*max_x, wx);
                *max_y = std::max(*max_y, wy);
                updated_any = true;
                continue;
            }

            // Risk field
            // Project onto velocity vector
            if (v_mag > 0.1)
            {
                double v_norm_x = vx / v_mag;
                double v_norm_y = vy / v_mag;
                
                double lon = dx * v_norm_x + dy * v_norm_y;
                double lat = dx * v_norm_y - dy * v_norm_x; // Perpendicular

                if (lon > 0 && lon < prediction_dist)
                {
                    // Anisotropic cost
                    // We want cost to be high near the obstacle and along the path
                    // Elliptical falloff?
                    // Effective distance
                    double lat_scale = radius; 
                    double lon_scale = prediction_dist; 
                    
                    // Normalize
                    double n_lon = lon / lon_scale;
                    double n_lat = lat / lat_scale;
                    
                    double dist_metric = n_lon*n_lon + n_lat*n_lat;
                    
                    if (dist_metric < 1.0)
                    {
                        double cost_val = 253.0 * (1.0 - dist_metric); // Linear falloff for simplicity
                        // Or exponential: 253 * exp(-k * dist_metric)
                        
                        unsigned char old_cost = costmap_[getIndex(i, j)];
                        unsigned char new_cost = (unsigned char)cost_val;
                        
                        if (new_cost > old_cost)
                        {
                            costmap_[getIndex(i, j)] = new_cost;
                            last_seen_timestamps_[getIndex(i, j)] = current_time;
                            
                            *min_x = std::min(*min_x, wx);
                            *min_y = std::min(*min_y, wy);
                            *max_x = std::max(*max_x, wx);
                            *max_y = std::max(*max_y, wy);
                            updated_any = true;
                        }
                    }
                }
            }
        }
    }
    
    if (updated_any)
    {
        ROS_INFO_THROTTLE(2.0, "DynamicObstacleLayer: Updated costmap for obstacle at (%.2f, %.2f)", cx, cy);
    }
    else
    {
        ROS_DEBUG_THROTTLE(2.0, "DynamicObstacleLayer: No cells updated for obstacle at (%.2f, %.2f) (Radius: %.2f)", cx, cy, radius);
    }
}

void DynamicObstacleLayer::updateCosts(costmap_2d::Costmap2D& master_grid, int min_i, int min_j, int max_i, int max_j)
{
  if (!enabled_) return;

  // Force updateWithMax to ensure we write to master
  updateWithMax(master_grid, min_i, min_j, max_i, max_j);

  // Debug: Check if updateCosts is actually called and what it does
  static bool first_call = true;
  if (first_call)
  {
      ROS_INFO("DynamicObstacleLayer: updateCosts called for the first time.");
      first_call = false;
  }

  // Debug: Check a specific point (1.0, 0.0) to see if it has cost in master_grid
  unsigned int mx, my;
  if (master_grid.worldToMap(1.0, 0.0, mx, my))
  {
      unsigned char cost = master_grid.getCost(mx, my);
      if (cost == LETHAL_OBSTACLE)
      {
          ROS_INFO_THROTTLE(2.0, "DynamicObstacleLayer: Master grid has LETHAL at (1.0, 0.0)");
      }
      else if (cost > 0)
      {
          ROS_INFO_THROTTLE(2.0, "DynamicObstacleLayer: Master grid has cost %d at (1.0, 0.0)", cost);
      }
      // else: FREE, do not log to reduce noise unless debugging specifically
  }
}

void DynamicObstacleLayer::reset()
{
    CostmapLayer::reset();
    std::fill(last_seen_timestamps_.begin(), last_seen_timestamps_.end(), 0.0);
    current_ = true;
}

}
