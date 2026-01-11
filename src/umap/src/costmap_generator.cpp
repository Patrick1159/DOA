#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>
#include <geometry_msgs/PoseStamped.h>
#include <tf2_geometry_msgs/tf2_geometry_msgs.h>
#include <umap/TrackerDetails.h>
#include <mutex>
#include <vector>
#include <cmath>
#include <tf2/LinearMath/Quaternion.h>
#include <tf2/LinearMath/Matrix3x3.h>

class CostmapGenerator
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber dync_sub_;
    ros::Subscriber pointcloud_sub_;
    ros::Subscriber map_sub_;
    ros::Subscriber goal_sub_;
    ros::Subscriber pose_sub_;
    ros::Publisher costmap_pub_;
    
    nav_msgs::OccupancyGrid static_map_;
    bool has_static_map_;
    
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // Parameters
    bool simulation_;
    double resolution_;
    double map_width_;
    double map_height_;
    double robot_height_;
    double obstacle_threshold_;
    double free_threshold_;
    double reach_tolerance_;
    double last_dync_info_time_;
    double age_tolerance_;
    double eps_;
    double t_pred_;
    double vel_uncertainty_rate_;
    std::string map_frame_;

    nav_msgs::OccupancyGrid costmap_;

    // Goal and Pose state
    geometry_msgs::PoseStamped current_goal_;
    bool has_goal_;
    geometry_msgs::PoseStamped current_pose_;
    bool has_pose_;

    // Dynamic obstacles data structures
    struct DynamicObstacle {
        int32_t id;
        geometry_msgs::Point position;
        geometry_msgs::Vector3 velocity;
        geometry_msgs::Vector3 dimensions;
        double last_update_time;
        double age;
    };
    
    std::vector<DynamicObstacle> dynamic_obstacles_;
    std::mutex obstacles_mutex_;  // Mutex for thread-safe access to dynamic_obstacles_

public:
    CostmapGenerator() : tf_listener_(tf_buffer_), has_static_map_(false), has_goal_(false), has_pose_(false)
    {
        // Load parameters
        nh_.param("simulation", simulation_, true);
        nh_.param("resolution", resolution_, 0.05);
        nh_.param("map_width", map_width_, 20.0);
        nh_.param("map_height", map_height_, 20.0);
        nh_.param("robot_height", robot_height_, 2.0);
        nh_.param("obstacle_threshold", obstacle_threshold_, 0.1);
        nh_.param("free_threshold", free_threshold_, 0.05);
        nh_.param("reach_tolerance", reach_tolerance_, 0.5);
        nh_.param("age_tolerance", age_tolerance_, 1.0);
        nh_.param("eps", eps_, 0.25);
        nh_.param("t_pred", t_pred_, 1.0);
        nh_.param("vel_uncertainty_rate", vel_uncertainty_rate_, 0.2);
        nh_.param<std::string>("map_frame", map_frame_, "map");
        
        // Initialize subscribers and publishers
        pointcloud_sub_ = nh_.subscribe("/velodyne_points", 1, &CostmapGenerator::pointCloudCallback, this);
        map_sub_ = nh_.subscribe("/map", 1, &CostmapGenerator::mapCallback, this);
        goal_sub_ = nh_.subscribe("/move_base_simple/goal", 1, &CostmapGenerator::goalCallback, this);
        pose_sub_ = nh_.subscribe("/camera/current_pose", 1, &CostmapGenerator::poseCallback, this);
        dync_sub_ = nh_.subscribe("/rematch/tracker_details", 1, &CostmapGenerator::dynamicObstaclesCallback, this);

        costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/costmap", 1);
        
        // Initialize costmap
        initializeCostmap();
        
        ROS_INFO("Costmap Generator initialized with eps=%.2f", eps_);
    }
    
    void initializeCostmap()
    {
        costmap_.header.frame_id = map_frame_;
        costmap_.info.resolution = resolution_;
        costmap_.info.width = static_cast<uint32_t>(map_width_ / resolution_);
        costmap_.info.height = static_cast<uint32_t>(map_height_ / resolution_);
        costmap_.info.origin.position.x = -map_width_ / 2.0;
        costmap_.info.origin.position.y = -map_height_ / 2.0;
        costmap_.info.origin.position.z = 0.0;
        costmap_.info.origin.orientation.w = 1.0;
        
        costmap_.data.resize(costmap_.info.width * costmap_.info.height, -1);
    }
    
    // Callback to receive static map
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
    {
        static_map_ = *msg;
        has_static_map_ = true;
        ROS_INFO_ONCE("Received static map");
    }

    void goalCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        current_goal_ = *msg;
        has_goal_ = true;
        ROS_INFO("Received goal. Clearing obstacles within %.2fm", reach_tolerance_);
    }

    void poseCallback(const geometry_msgs::PoseStamped::ConstPtr& msg)
    {
        current_pose_ = *msg;
        has_pose_ = true;
    }

    void clearCircularArea(const geometry_msgs::PoseStamped& target_pose)
    {
        geometry_msgs::PoseStamped map_pose;
        try
        {
            if (target_pose.header.frame_id != map_frame_)
            {
                map_pose = tf_buffer_.transform(target_pose, map_frame_, ros::Duration(0.1));
            }
            else
            {
                map_pose = target_pose;
            }
        }
        catch (tf2::TransformException& ex)
        {
            ROS_WARN_THROTTLE(1.0, "Cannot transform pose to map frame: %s", ex.what());
            return;
        }

        double gx = map_pose.pose.position.x;
        double gy = map_pose.pose.position.y;

        int cx = static_cast<int>((gx - costmap_.info.origin.position.x) / resolution_);
        int cy = static_cast<int>((gy - costmap_.info.origin.position.y) / resolution_);
        
        int radius_cells = static_cast<int>(reach_tolerance_ / resolution_);
        int radius_sq = radius_cells * radius_cells;

        for (int dy = -radius_cells; dy <= radius_cells; ++dy)
        {
            for (int dx = -radius_cells; dx <= radius_cells; ++dx)
            {
                if (dx * dx + dy * dy > radius_sq) continue;

                int nx = cx + dx;
                int ny = cy + dy;

                if (nx >= 0 && nx < static_cast<int>(costmap_.info.width) &&
                    ny >= 0 && ny < static_cast<int>(costmap_.info.height))
                {
                    int index = ny * costmap_.info.width + nx;
                    costmap_.data[index] = 0; // Force free
                }
            }
        }
    }
    

    void dynamicObstaclesCallback(const umap::TrackerDetails::ConstPtr& msg)
    {
        last_dync_info_time_ = ros::Time::now().toSec();
        std::lock_guard<std::mutex> lock(obstacles_mutex_);

        dynamic_obstacles_.clear();
        size_t num_obstacles = std::min(
            std::min(msg->tracker_ids.size(), msg->positions.size()),
            std::min(msg->velocities.size(), msg->box_dimensions.size())
        );

        // 获取相机到odom的变换
        geometry_msgs::TransformStamped camera_to_odom_tf;
        try {
            if (simulation_) {
                camera_to_odom_tf = tf_buffer_.lookupTransform(
                    "odom",  // target frame: odom
                    "camera_depth_frame",  // source frame: camera_depth_frame
                    ros::Time(0)
                );
            }
            else{
            camera_to_odom_tf = tf_buffer_.lookupTransform(
                "odom",  // target frame: odom
                "real_baselink",  // source frame: real_baselink
                ros::Time(0)
            );
            }
        } catch (tf2::TransformException& ex) {
            ROS_WARN_THROTTLE(1.0, "Cannot get camera to odom transform: %s", ex.what());
            return;
        }
        
        tf2::Transform camera_to_odom;
        tf2::fromMsg(camera_to_odom_tf.transform, camera_to_odom);
        
        // correction
        tf2::Quaternion q_fix;
        q_fix.setRPY(0, 0, M_PI / 2.0);  // roll=0, pitch=0, yaw=+90°
        tf2::Transform T_fix;
        T_fix.setIdentity();
        T_fix.setRotation(q_fix);

        tf2::Transform camera_to_odom_corrected;
        if (simulation_) {
            // In simulation, no correction needed
            camera_to_odom_corrected = camera_to_odom;
        } else {
            camera_to_odom_corrected = T_fix * camera_to_odom;
        }

        for (size_t i = 0; i < num_obstacles; ++i)
        {
            DynamicObstacle obstacle;
            obstacle.id = msg->tracker_ids[i];
            obstacle.age = msg->ages[i];
            if (obstacle.age > age_tolerance_) {
                ROS_INFO("Ignoring obstacle %d due to age %.2f > %.2f", obstacle.id, obstacle.age, age_tolerance_);
                continue;
            }           
            // 1. 从相机坐标系转换到baselink坐标系
            tf2::Vector3 pos_camera(
                msg->positions[i].x,
                msg->positions[i].y,
                msg->positions[i].z
            );
            
            tf2::Vector3 vel_camera(
                msg->velocities[i].x,
                msg->velocities[i].y,
                msg->velocities[i].z
            );

            // 位置变换: 应用相机到odom的变换
            tf2::Vector3 pos_odom;
            tf2::Vector3 vel_odom;
            if (simulation_) {
                // In simulation, no correction needed
                ROS_INFO("Simulation mode: no correction applied to dynamic obstacles");
                pos_odom = pos_camera;
                vel_odom = vel_camera;
            } else {
                ROS_INFO("Real mode: applying correction to dynamic obstacles");
                pos_odom = camera_to_odom_corrected * pos_camera;
                vel_odom = camera_to_odom_corrected.getBasis() * vel_camera;
            }
            
            // 保存转换后的位置和速度
            obstacle.position.x = pos_odom.x();
            obstacle.position.y = pos_odom.y();
            obstacle.position.z = pos_odom.z();
            
            obstacle.velocity.x = vel_odom.x();
            obstacle.velocity.y = vel_odom.y();
            obstacle.velocity.z = vel_odom.z();
            
            // 尺寸保持不变（标量）
            obstacle.dimensions = msg->box_dimensions[i];
            
            if (i < msg->last_update_times.size()) {
                obstacle.last_update_time = msg->last_update_times[i];
            }
            if (i < msg->ages.size()) {
                obstacle.age = msg->ages[i];
            }
            
            dynamic_obstacles_.push_back(obstacle);
            
            ROS_INFO("Dynamic obstacle %d: camera (%.2f, %.2f) -> odom (%.2f, %.2f)",
                    obstacle.id,
                    msg->positions[i].x, msg->positions[i].y,
                    obstacle.position.x, obstacle.position.y);
        }
        
        ROS_DEBUG("Received %zu dynamic obstacles", dynamic_obstacles_.size());
    }
    
    void markDynamicObstaclesOnCostmap()
    {
        std::lock_guard<std::mutex> lock(obstacles_mutex_);

        if (last_dync_info_time_ + 0.1 < ros::Time::now().toSec()) {
            ROS_WARN_THROTTLE(0.1, "No recent dynamic obstacle info received");
            return;
        }
        int width  = costmap_.info.width;
        int height = costmap_.info.height;

        std::vector<int8_t> dynamic_layer_(width * height, 0);
        std::fill(dynamic_layer_.begin(), dynamic_layer_.end(), 0);

        for (const auto& obstacle : dynamic_obstacles_)
        {
            // Convert world coordinates to grid coordinates
            double wx = obstacle.position.x;
            double wy = obstacle.position.y;

            double ox = costmap_.info.origin.position.x;
            double oy = costmap_.info.origin.position.y;

            int mx = static_cast<int>(std::floor((wx - ox) / resolution_));
            int my = static_cast<int>(std::floor((wy - oy) / resolution_));

            if (mx < 0 || mx >= width || my < 0 || my >= height)
                continue;

            int radius = std::max(1,
                static_cast<int>(std::ceil(eps_ / resolution_))
            );
            int r_sq = radius * radius;

            for (int dy = -radius; dy <= radius; ++dy) {
                for (int dx = -radius; dx <= radius; ++dx) {
                    if (dx*dx + dy*dy > r_sq) continue;

                    int nx = mx + dx;
                    int ny = my + dy;

                    if (nx < 0 || nx >= width || ny < 0 || ny >= height)
                        continue;

                    dynamic_layer_[ny * width + nx] = 100;
                }
            }
            
            // Optional: Inflate based on velocity magnitude
            double t_pred = t_pred_; // seconds
            double vx = obstacle.velocity.x;
            double vy = obstacle.velocity.y;

            int N = 20;

            double base_radius = eps_;        // m
            double uncertainty_rate = vel_uncertainty_rate_;    // m/s

            for (int i = 0; i <= N; ++i) {
                double alpha = static_cast<double>(i) / N;
                double t = alpha * t_pred;

                // 预测位置（map cell）
                double px = mx + (vx * t) / resolution_;
                double py = my + (vy * t) / resolution_;

                // 时间相关的不确定性半径
                double radius = base_radius + uncertainty_rate * t;
                int radius_cells = static_cast<int>(radius / resolution_);

                for (int dx = -radius_cells; dx <= radius_cells; ++dx) {
                    for (int dy = -radius_cells; dy <= radius_cells; ++dy) {
                        if (dx*dx + dy*dy > radius_cells*radius_cells)
                            continue;

                        int cx = static_cast<int>(px) + dx;
                        int cy = static_cast<int>(py) + dy;

                        if (cx < 0 || cx >= width || cy < 0 || cy >= height)
                            continue;

                        int8_t cost = static_cast<int8_t>(100); // * (1.0 - alpha);
                        dynamic_layer_[cy * width + cx] =
                            std::max(dynamic_layer_[cy * width + cx], cost);
                    }
                }
            }
        }
        
        for (int i = 0; i < width * height; ++i) 
        {
            costmap_.data[i] = std::max(costmap_.data[i], dynamic_layer_[i]);
        }
    }

    void pointCloudCallback(const sensor_msgs::PointCloud2::ConstPtr& msg)
    {
        sensor_msgs::PointCloud2 transformed_cloud;
        
        // Transform point cloud to map frame if necessary
        if (msg->header.frame_id != map_frame_)
        {
            if (!transformPointCloud(*msg, transformed_cloud))
            {
                return;
            }
        }
        else
        {
            transformed_cloud = *msg;
        }
        
        // Clear previous costmap data
        if (has_static_map_)
        {
            for (unsigned int y = 0; y < costmap_.info.height; ++y)
            {
                for (unsigned int x = 0; x < costmap_.info.width; ++x)
                {
                    int index = y * costmap_.info.width + x;
                    
                    double wx = costmap_.info.origin.position.x + (x + 0.5) * resolution_;
                    double wy = costmap_.info.origin.position.y + (y + 0.5) * resolution_;
                    
                    int mx = static_cast<int>((wx - static_map_.info.origin.position.x) / static_map_.info.resolution);
                    int my = static_cast<int>((wy - static_map_.info.origin.position.y) / static_map_.info.resolution);
                    
                    if (mx >= 0 && mx < static_cast<int>(static_map_.info.width) &&
                        my >= 0 && my < static_cast<int>(static_map_.info.height))
                    {
                        int map_index = my * static_map_.info.width + mx;
                        if (static_map_.data[map_index] == 100)
                            costmap_.data[index] = 100;
                        else
                            costmap_.data[index] = 0;
                    }
                    else
                    {
                        costmap_.data[index] = 0;
                    }
                }
            }
        }
        else
        {
            std::fill(costmap_.data.begin(), costmap_.data.end(), 0);
        }
        
        // Generate costmap from point cloud
        generateCostmap(transformed_cloud);
        
        // Mark dynamic obstacles on the costmap
        markDynamicObstaclesOnCostmap();
        
        if (has_goal_) clearCircularArea(current_goal_);
        if (has_pose_) clearCircularArea(current_pose_);

        // Publish costmap
        costmap_.header.stamp = ros::Time::now();
        costmap_pub_.publish(costmap_);
        ROS_DEBUG_THROTTLE(1.0, "Published costmap with dynamic obstacles");
    }
    
    bool transformPointCloud(const sensor_msgs::PointCloud2& input_cloud, 
                           sensor_msgs::PointCloud2& output_cloud)
    {
        try
        {
            geometry_msgs::TransformStamped transform = tf_buffer_.lookupTransform(
                map_frame_, input_cloud.header.frame_id, ros::Time(0), ros::Duration(1.0));
            
            tf2::doTransform(input_cloud, output_cloud, transform);
            output_cloud.header.stamp = transform.header.stamp;
            
            return true;
        }
        catch (tf2::TransformException& ex)
        {
            ROS_WARN("Transform failed: %s", ex.what());
            return false;
        }
    }
    
    void generateCostmap(const sensor_msgs::PointCloud2& cloud)
    {
        // Create iterators for x, y, z coordinates
        sensor_msgs::PointCloud2ConstIterator<float> iter_x(cloud, "x");
        sensor_msgs::PointCloud2ConstIterator<float> iter_y(cloud, "y");
        sensor_msgs::PointCloud2ConstIterator<float> iter_z(cloud, "z");
        
        // Process each point
        for (; iter_x != iter_x.end(); ++iter_x, ++iter_y, ++iter_z)
        {
            float x = *iter_x;
            float y = *iter_y;
            float z = *iter_z;
            
            // Skip invalid points (NaN or infinity)
            if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z))
            {
                continue;
            }
            
            // Skip points outside map bounds
            if (x < costmap_.info.origin.position.x || 
                x > costmap_.info.origin.position.x + map_width_ ||
                y < costmap_.info.origin.position.y || 
                y > costmap_.info.origin.position.y + map_height_)
            {
                continue;
            }
            
            // Skip points too high (above robot)
            if (z > robot_height_)
            {
                continue;
            }
            
            // Convert world coordinates to grid coordinates
            int grid_x = static_cast<int>((x - costmap_.info.origin.position.x) / resolution_);
            int grid_y = static_cast<int>((y - costmap_.info.origin.position.y) / resolution_);
            
            // Check bounds
            if (grid_x >= 0 && grid_x < static_cast<int>(costmap_.info.width) &&
                grid_y >= 0 && grid_y < static_cast<int>(costmap_.info.height))
            {
                int index = grid_y * costmap_.info.width + grid_x;
                
                // Mark as obstacle if point is high enough
                if (z > obstacle_threshold_)
                {
                    costmap_.data[index] = 100;  // Occupied
                }
                else if (costmap_.data[index] != 100)
                {
                    costmap_.data[index] = 0;   // Free space
                }
            }
        }
        
        // Apply inflation around obstacles
        inflateObstacles();
    }
    
    void inflateObstacles()
    {
        std::vector<int8_t> inflated_map = costmap_.data;
        int inflation_radius = static_cast<int>(0.3 / resolution_); // 30cm inflation
        
        for (int y = 0; y < static_cast<int>(costmap_.info.height); ++y)
        {
            for (int x = 0; x < static_cast<int>(costmap_.info.width); ++x)
            {
                int index = y * costmap_.info.width + x;
                
                if (costmap_.data[index] == 100) // If occupied
                {
                    // Inflate around this obstacle
                    for (int dy = -inflation_radius; dy <= inflation_radius; ++dy)
                    {
                        for (int dx = -inflation_radius; dx <= inflation_radius; ++dx)
                        {
                            int nx = x + dx;
                            int ny = y + dy;
                            
                            if (nx >= 0 && nx < static_cast<int>(costmap_.info.width) &&
                                ny >= 0 && ny < static_cast<int>(costmap_.info.height))
                            {
                                int n_index = ny * costmap_.info.width + nx;
                                double distance = sqrt(dx * dx + dy * dy) * resolution_;
                                
                                if (distance <= 0.3 && inflated_map[n_index] != 100)
                                {
                                    // Linear cost decay
                                    int cost = static_cast<int>(99 * (1.0 - distance / 0.3));
                                    inflated_map[n_index] = std::max(inflated_map[n_index], static_cast<int8_t>(cost));
                                }
                            }
                        }
                    }
                }
            }
        }
        
        costmap_.data = inflated_map;
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "costmap_generator");
    
    CostmapGenerator generator;
    
    ros::spin();
    
    return 0;
}
