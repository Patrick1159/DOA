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
    ros::Subscriber map_sub_;
    ros::Publisher costmap_pub_;
    
    nav_msgs::OccupancyGrid static_map_;
    bool has_static_map_;
    
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
    
    nav_msgs::OccupancyGrid costmap_;

public:
    CostmapGenerator() : tf_listener_(tf_buffer_), has_static_map_(false)
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
        
        // Initialize subscribers and publishers
        pointcloud_sub_ = nh_.subscribe("/velodyne_points", 1, &CostmapGenerator::pointCloudCallback, this);
        map_sub_ = nh_.subscribe("/map", 1, &CostmapGenerator::mapCallback, this);
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
    
    void mapCallback(const nav_msgs::OccupancyGrid::ConstPtr& msg)
    {
        static_map_ = *msg;
        has_static_map_ = true;
        ROS_INFO_ONCE("Received static map");
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
