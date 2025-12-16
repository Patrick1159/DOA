#include <ros/ros.h>
#include <sensor_msgs/PointCloud2.h>
#include <nav_msgs/OccupancyGrid.h>
#include <sensor_msgs/point_cloud2_iterator.h>
#include <tf2_ros/transform_listener.h>
#include <tf2_sensor_msgs/tf2_sensor_msgs.h>
#include <geometry_msgs/TransformStamped.h>

class CostmapGenerator
{
private:
    ros::NodeHandle nh_;
    ros::Subscriber pointcloud_sub_;
    ros::Publisher costmap_pub_;
    
    tf2_ros::Buffer tf_buffer_;
    tf2_ros::TransformListener tf_listener_;
    
    // Parameters
    double resolution_;
    double map_width_;
    double map_height_;
    double robot_height_;
    double obstacle_threshold_;
    double free_threshold_;
    std::string map_frame_;
    std::string pointcloud_frame_;
    // Inflation and clearance control
    double inflation_radius_m_;
    double inflation_decay_alpha_;
    // Border handling
    double border_band_m_;
    int8_t border_cost_;
    // Unknown handling
    bool unknown_is_costly_;
    int8_t unknown_fill_cost_;
    
    nav_msgs::OccupancyGrid costmap_;

public:
    CostmapGenerator() : tf_listener_(tf_buffer_)
    {
        // Load parameters
        nh_.param("resolution", resolution_, 0.05);
        nh_.param("map_width", map_width_, 20.0);
        nh_.param("map_height", map_height_, 20.0);
        nh_.param("robot_height", robot_height_, 2.0);
        nh_.param("obstacle_threshold", obstacle_threshold_, 0.1);
        nh_.param("free_threshold", free_threshold_, 0.05);
        nh_.param<std::string>("map_frame", map_frame_, "map");
        nh_.param<std::string>("pointcloud_frame", pointcloud_frame_, "velodyne");
        // Tunables to reduce edge-hugging
        nh_.param("inflation_radius_m", inflation_radius_m_, 0.6);
        nh_.param("inflation_decay_alpha", inflation_decay_alpha_, 8.0);
        nh_.param("border_band_m", border_band_m_, 0.4);
        int border_cost_tmp;
        nh_.param("border_cost", border_cost_tmp, 90);
        border_cost_ = static_cast<int8_t>(std::max(0, std::min(border_cost_tmp, 100)));
        nh_.param("unknown_is_costly", unknown_is_costly_, true);
        int unknown_cost_tmp;
        nh_.param("unknown_fill_cost", unknown_cost_tmp, 50);
        unknown_fill_cost_ = static_cast<int8_t>(std::max(0, std::min(unknown_cost_tmp, 100)));
        
        // Initialize subscribers and publishers
        pointcloud_sub_ = nh_.subscribe("/velodyne_points", 1, &CostmapGenerator::pointCloudCallback, this);
        costmap_pub_ = nh_.advertise<nav_msgs::OccupancyGrid>("/costmap", 1);
        
        // Initialize costmap
        initializeCostmap();
        
        ROS_INFO("Costmap Generator initialized");
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
        
        // Reset previous costmap data: keep unknowns by default
        std::fill(costmap_.data.begin(), costmap_.data.end(), -1);
        
        // Generate costmap from point cloud
        generateCostmap(transformed_cloud);
        
        // Apply high-cost border band to discourage edge-hugging
        applyBorderBand();

        // Publish costmap
        costmap_.header.stamp = ros::Time::now();
        costmap_pub_.publish(costmap_);
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

        // Optionally assign a medium cost to remaining unknown cells to avoid planner treating them as cheap
        if (unknown_is_costly_)
        {
            for (size_t i = 0; i < costmap_.data.size(); ++i)
            {
                if (costmap_.data[i] == -1)
                {
                    costmap_.data[i] = unknown_fill_cost_;
                }
            }
        }
    }
    
    void inflateObstacles()
    {
        std::vector<int8_t> inflated_map = costmap_.data;
        int inflation_radius = static_cast<int>(inflation_radius_m_ / resolution_);
        
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
                                
                                if (distance <= inflation_radius_m_ && inflated_map[n_index] != 100)
                                {
                                    // Exponential cost decay: high near obstacles, drops quickly with distance
                                    double decay = std::exp(-inflation_decay_alpha_ * (distance / std::max(1e-6, inflation_radius_m_)));
                                    int cost = static_cast<int>(std::round(99.0 * decay));
                                    cost = std::max(0, std::min(cost, 100));
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

    void applyBorderBand()
    {
        int band_cells = static_cast<int>(border_band_m_ / resolution_);
        if (band_cells <= 0) return;

        int width = static_cast<int>(costmap_.info.width);
        int height = static_cast<int>(costmap_.info.height);

        for (int y = 0; y < height; ++y)
        {
            for (int x = 0; x < width; ++x)
            {
                bool in_band = (x < band_cells) || (x >= width - band_cells) ||
                               (y < band_cells) || (y >= height - band_cells);
                if (in_band)
                {
                    int idx = y * width + x;
                    if (costmap_.data[idx] != 100)
                    {
                        costmap_.data[idx] = std::max(costmap_.data[idx], border_cost_);
                    }
                }
            }
        }
    }
};

int main(int argc, char** argv)
{
    ros::init(argc, argv, "costmap_generator");
    
    CostmapGenerator generator;
    
    ros::spin();
    
    return 0;
}